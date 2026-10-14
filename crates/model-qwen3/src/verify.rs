use std::path::Path;

use engine_core::{FixtureManifest, SafeTensorCatalog, Tolerance};

use crate::weights::{LayerWeights, load_tensor};
use crate::{ModelError, Qwen3Artifacts, Qwen3Config, Result};

// Each fixture tensor is listed with the manifest tolerance that applies to it.
const PREFILL_FILE: &str = "block_0.safetensors";
const PREFILL_TENSORS: [(&str, &str); 12] = [
    ("input_norm", "norm"),
    ("q", "projection"),
    ("k", "projection"),
    ("v", "projection"),
    ("q_rope", "rope"),
    ("k_rope", "rope"),
    ("scores", "scores"),
    ("attention_probs", "probs"),
    ("attention_output", "attention_output"),
    ("mlp_input", "norm"),
    ("mlp_output", "mlp_output"),
    ("output", "block_output"),
];
const DECODE_FILE: &str = "decode_block_0.safetensors";
const DECODE_TENSORS: [(&str, &str); 16] = [
    ("input_norm", "norm"),
    ("q", "projection"),
    ("k", "projection"),
    ("v", "projection"),
    ("q_rope", "rope"),
    ("k_rope", "rope"),
    // cached keys are rotated keys, and cached values are projected values
    ("cache_k_in", "rope"),
    ("cache_v_in", "projection"),
    ("cache_k_out", "rope"),
    ("cache_v_out", "projection"),
    ("scores", "scores"),
    ("attention_probs", "probs"),
    ("attention_output", "attention_output"),
    ("mlp_input", "norm"),
    ("mlp_output", "mlp_output"),
    ("output", "block_output"),
];
const GENERATION_FILE: &str = "generation.safetensors";

/// How closely one GPU tensor matched its fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryResult {
    pub name: String,
    pub max_abs_error: f32,
    pub tolerance: Tolerance,
}

/// The tensors recorded for one pass through layer 0.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct LayerFixture {
    pub name: &'static str,
    pub rows: usize,
    /// The layer input, `[rows, hidden_size]`.
    pub input: Vec<f32>,
    pub expected: Vec<ExpectedTensor>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct ExpectedTensor {
    pub name: &'static str,
    pub values: Vec<f32>,
    pub tolerance: Tolerance,
}

/// The tensors recorded for the fixture prompt's pass through the whole
/// model, and for the decode step after it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct ModelFixture {
    pub token_ids: Vec<u32>,
    /// The prompt's embeddings, `[tokens, hidden_size]`.
    pub embedding: Vec<f32>,
    /// The last prompt token after each layer, `[layers, hidden_size]`.
    pub layer_outputs: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub prefill_logits: Vec<f32>,
    /// The first generated token, which the decode step runs.
    pub decode_token_id: u32,
    pub decode_logits: Vec<f32>,
    pub layer_tolerance: Tolerance,
    pub logits_tolerance: Tolerance,
}

/// Runs layer 0 on the GPU with the real weights and compares every recorded
/// tensor with the golden fixtures.
///
/// The prompt is run first, then one more token is decoded against the cache
/// that run left behind. Both passes are checked through the whole layer.
///
/// # Errors
///
/// Returns an error when a fixture or weight cannot be loaded, the GPU is
/// unavailable, or a tensor is outside its fixture tolerance.
pub fn verify_layer_0(
    artifacts: &Qwen3Artifacts,
    fixture_dir: impl AsRef<Path>,
) -> Result<Vec<BoundaryResult>> {
    let fixture_dir = fixture_dir.as_ref();
    let config = &artifacts.config;
    let manifest = FixtureManifest::load(fixture_dir.join("manifest.json"))?;
    manifest.validate_files(fixture_dir)?;

    let weights = LayerWeights::load(artifacts, 0)?;
    let prefill = load_fixture(
        &manifest,
        fixture_dir,
        config,
        "prefill",
        PREFILL_FILE,
        &PREFILL_TENSORS,
    )?;
    let decode = load_fixture(
        &manifest,
        fixture_dir,
        config,
        "decode",
        DECODE_FILE,
        &DECODE_TENSORS,
    )?;

    run_on_gpu(config, &weights, &prefill, &decode)
}

/// Runs the fixture prompt through the whole model on the GPU, starting from
/// its token IDs, and compares the embeddings, the last token after each
/// layer, the final norm and the logits with the golden fixtures.
///
/// It then decodes the first generated token against the caches the prompt
/// left in all 28 layers and compares those logits too.
///
/// # Errors
///
/// Returns an error when a fixture or weight cannot be loaded, the GPU is
/// unavailable, or a tensor is outside its fixture tolerance.
pub fn verify_model(
    artifacts: &Qwen3Artifacts,
    fixture_dir: impl AsRef<Path>,
) -> Result<Vec<BoundaryResult>> {
    let fixture_dir = fixture_dir.as_ref();
    let config = &artifacts.config;
    let manifest = FixtureManifest::load(fixture_dir.join("manifest.json"))?;
    manifest.validate_files(fixture_dir)?;

    let rows = manifest.token_ids.len();
    let prefill = SafeTensorCatalog::open_files([fixture_dir.join(PREFILL_FILE)])?;
    let generation = SafeTensorCatalog::open_files([fixture_dir.join(GENERATION_FILE)])?;

    let fixture = ModelFixture {
        token_ids: manifest.token_ids.clone(),
        embedding: load_tensor(&prefill, "input", &[rows, config.hidden_size])?,
        layer_outputs: load_tensor(
            &generation,
            "layer_outputs",
            &[config.num_hidden_layers, config.hidden_size],
        )?,
        final_norm: load_tensor(&generation, "final_norm", &[config.hidden_size])?,
        prefill_logits: load_tensor(&generation, "prefill_logits", &[config.vocab_size])?,
        decode_token_id: manifest
            .expected_greedy_token_ids
            .first()
            .copied()
            .ok_or_else(|| ModelError::MissingTensor("expected_greedy_token_ids".to_owned()))?,
        decode_logits: load_tensor(&generation, "decode_logits", &[config.vocab_size])?,
        layer_tolerance: manifest.tolerance(GENERATION_FILE, "layer_output")?,
        logits_tolerance: manifest.tolerance(GENERATION_FILE, "logits")?,
    };

    run_model_on_gpu(artifacts, &fixture)
}

fn load_fixture(
    manifest: &FixtureManifest,
    fixture_dir: &Path,
    config: &Qwen3Config,
    name: &'static str,
    file: &str,
    tensors: &[(&'static str, &str)],
) -> Result<LayerFixture> {
    let catalog = SafeTensorCatalog::open_files([fixture_dir.join(file)])?;
    let rows = catalog.load("input")?.descriptor.layout.element_count() / config.hidden_size;

    let mut expected = Vec::with_capacity(tensors.len());
    for &(tensor, tolerance) in tensors {
        expected.push(ExpectedTensor {
            name: tensor,
            values: catalog.load(tensor)?.to_f32_vec()?,
            tolerance: manifest.tolerance(file, tolerance)?,
        });
    }

    Ok(LayerFixture {
        name,
        rows,
        input: load_tensor(&catalog, "input", &[rows, config.hidden_size])?,
        expected,
    })
}

#[cfg(feature = "cuda")]
fn run_on_gpu(
    config: &Qwen3Config,
    weights: &LayerWeights,
    prefill: &LayerFixture,
    decode: &LayerFixture,
) -> Result<Vec<BoundaryResult>> {
    use crate::block::Block;
    use crate::trace::Trace;

    let backend = cuda_backend::Backend::new()?;
    let block = Block::new(&backend, config, weights)?;
    let mut results = Vec::new();

    let mut trace = Trace::recording();
    let x = backend.upload(&prefill.input, &[prefill.rows, config.hidden_size])?;
    let (_, cache) = block.forward(&x, prefill.rows, None, &mut trace)?;
    compare(prefill, &trace, &mut results)?;

    // decode attends to the cache that prefill just built
    let mut trace = Trace::recording();
    let x = backend.upload(&decode.input, &[decode.rows, config.hidden_size])?;
    block.forward(&x, decode.rows, Some(&cache), &mut trace)?;
    compare(decode, &trace, &mut results)?;

    Ok(results)
}

#[cfg(feature = "cuda")]
fn compare(
    fixture: &LayerFixture,
    trace: &crate::trace::Trace,
    results: &mut Vec<BoundaryResult>,
) -> Result<()> {
    for expected in &fixture.expected {
        let name = format!("{}_{}", fixture.name, expected.name);
        let actual = trace.get(expected.name)?;
        let max_abs_error = expected
            .tolerance
            .compare(&name, actual, &expected.values)?;

        results.push(BoundaryResult {
            name,
            max_abs_error,
            tolerance: expected.tolerance,
        });
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_on_gpu(
    _config: &Qwen3Config,
    _weights: &LayerWeights,
    _prefill: &LayerFixture,
    _decode: &LayerFixture,
) -> Result<Vec<BoundaryResult>> {
    Err(cuda_backend::BackendError::CudaDisabled.into())
}

#[cfg(feature = "cuda")]
fn run_model_on_gpu(
    artifacts: &Qwen3Artifacts,
    fixture: &ModelFixture,
) -> Result<Vec<BoundaryResult>> {
    use crate::model::Model;
    use crate::trace::Trace;

    let hidden_size = artifacts.config.hidden_size;
    let rows = fixture.token_ids.len();
    let backend = cuda_backend::Backend::new()?;
    let model = Model::load(&backend, artifacts)?;

    let mut trace = Trace::recording();
    let (_, caches) = model.forward(&fixture.token_ids, &[], &mut trace)?;

    let mut results = Vec::new();
    // widening bfloat16 to float32 is exact, so the rows must match exactly
    check(
        &mut results,
        "embedding".to_owned(),
        trace.get("embedding")?,
        &fixture.embedding,
        Tolerance {
            atol: 0.0,
            rtol: 0.0,
        },
    )?;
    for (layer, expected) in fixture.layer_outputs.chunks(hidden_size).enumerate() {
        let name = format!("layer_{layer}_output");
        let output = trace.get(&name)?;
        let last_row = &output[(rows - 1) * hidden_size..];
        check(
            &mut results,
            name,
            last_row,
            expected,
            fixture.layer_tolerance,
        )?;
    }
    // the final norm has no tolerance of its own, so it is held to the layer one
    check(
        &mut results,
        "final_norm".to_owned(),
        trace.get("final_norm")?,
        &fixture.final_norm,
        fixture.layer_tolerance,
    )?;
    check(
        &mut results,
        "prefill_logits".to_owned(),
        trace.get("logits")?,
        &fixture.prefill_logits,
        fixture.logits_tolerance,
    )?;

    // decode attends to the caches that prefill just built in every layer
    let (logits, _) = model.forward(&[fixture.decode_token_id], &caches, &mut Trace::off())?;
    check(
        &mut results,
        "decode_logits".to_owned(),
        &backend.download(&logits)?,
        &fixture.decode_logits,
        fixture.logits_tolerance,
    )?;
    Ok(results)
}

#[cfg(feature = "cuda")]
fn check(
    results: &mut Vec<BoundaryResult>,
    name: String,
    actual: &[f32],
    expected: &[f32],
    tolerance: Tolerance,
) -> Result<()> {
    let max_abs_error = tolerance.compare(&name, actual, expected)?;
    results.push(BoundaryResult {
        name,
        max_abs_error,
        tolerance,
    });
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_model_on_gpu(
    _artifacts: &Qwen3Artifacts,
    _fixture: &ModelFixture,
) -> Result<Vec<BoundaryResult>> {
    Err(cuda_backend::BackendError::CudaDisabled.into())
}
