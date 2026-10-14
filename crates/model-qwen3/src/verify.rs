use std::path::Path;

use engine_core::{FixtureManifest, SafeTensorCatalog, Tolerance};

use crate::weights::{LayerWeights, load_tensor};
use crate::{ModelError, Qwen3Artifacts, Qwen3Config, RegressionPromptSet, Result};

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
// spare cache rows that never get written. they start as NaN, so reading one
// by mistake fails the checks
#[cfg(feature = "cuda")]
const SPARE_CACHE_ROWS: usize = 3;

/// How closely one GPU tensor matched its fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryResult {
    pub name: String,
    pub max_abs_error: f32,
    pub tolerance: Tolerance,
}

/// The tokens one prompt generated on the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationResult {
    pub prompt: String,
    /// The tokens that would stop generation early.
    pub eos_ids: Vec<u32>,
    pub token_ids: Vec<u32>,
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
            .ok_or(ModelError::EmptyManifestField("expected_greedy_token_ids"))?,
        decode_logits: load_tensor(&generation, "decode_logits", &[config.vocab_size])?,
        layer_tolerance: manifest.tolerance(GENERATION_FILE, "layer_output")?,
        logits_tolerance: manifest.tolerance(GENERATION_FILE, "logits")?,
    };

    run_model_on_gpu(artifacts, &fixture)
}

/// Loads the model on the GPU and greedily generates from every regression
/// prompt, checking that each produces the tokens the reference did.
///
/// One more run of the first prompt treats its second expected token as the
/// end-of-sequence token, and checks that generation stops right after it.
/// Every run is repeated without caches, and both must pick the same tokens.
/// The decode steps of every cached run must not allocate any buffers.
///
/// # Errors
///
/// Returns an error when a weight cannot be loaded, the GPU is unavailable,
/// there is no first prompt with at least two expected tokens, any run
/// generates different tokens, the cached and uncached runs disagree, or a
/// decode step allocates.
pub fn verify_generation(
    artifacts: &Qwen3Artifacts,
    prompts: &RegressionPromptSet,
) -> Result<Vec<GenerationResult>> {
    let first_expected = prompts
        .cases
        .first()
        .map_or(0, |case| case.expected_token_ids.len());
    if first_expected < 2 {
        return Err(ModelError::InvalidConfig(
            "the first regression prompt needs at least two expected tokens".to_owned(),
        ));
    }
    generate_on_gpu(artifacts, prompts)
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
    use crate::kv_cache::KvCache;
    use crate::trace::Trace;
    use crate::workspace::Workspace;

    let backend = cuda_backend::Backend::new()?;
    let block = Block::new(&backend, config, weights)?;
    let capacity = prefill.rows + decode.rows + SPARE_CACHE_ROWS;
    let mut cache = KvCache::filled_with_nan(&backend, config, capacity)?;
    let mut results = Vec::new();

    let mut trace = Trace::recording();
    let mut workspace = Workspace::new(&backend, config, prefill.rows, capacity)?;
    workspace.x = backend.upload(&prefill.input, &[prefill.rows, config.hidden_size])?;
    block.forward(&mut workspace, &mut cache, &mut trace)?;
    compare(prefill, &trace, &mut results)?;

    // decode attends to the cache that prefill just filled
    let mut trace = Trace::recording();
    let mut workspace = Workspace::new(&backend, config, decode.rows, capacity)?;
    workspace.x = backend.upload(&decode.input, &[decode.rows, config.hidden_size])?;
    block.forward(&mut workspace, &mut cache, &mut trace)?;
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
    use crate::kv_cache::KvCache;
    use crate::model::Model;
    use crate::trace::Trace;

    let config = &artifacts.config;
    let hidden_size = config.hidden_size;
    let rows = fixture.token_ids.len();
    let backend = cuda_backend::Backend::new()?;
    let model = Model::load(&backend, artifacts)?;

    // room for the prompt and the decoded token, plus the spare rows
    let capacity = rows + 1 + SPARE_CACHE_ROWS;
    let mut caches = (0..config.num_hidden_layers)
        .map(|_| KvCache::filled_with_nan(&backend, config, capacity))
        .collect::<Result<Vec<_>>>()?;

    let mut trace = Trace::recording();
    let mut workspace = model.new_workspace(rows, capacity)?;
    model.prefill(&fixture.token_ids, &mut caches, &mut workspace, &mut trace)?;

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

    // decode attends to the caches that prefill just filled in every layer
    let mut workspace = model.new_workspace(1, capacity)?;
    let logits = model.decode_step(
        fixture.decode_token_id,
        &mut caches,
        &mut workspace,
        &mut Trace::off(),
    )?;
    check(
        &mut results,
        "decode_logits".to_owned(),
        logits,
        &fixture.decode_logits,
        fixture.logits_tolerance,
    )?;

    // more decode steps fill the spare rows, reusing the same workspace
    let allocations_before = backend.allocations();
    for _ in 0..SPARE_CACHE_ROWS {
        model.decode_step(
            fixture.decode_token_id,
            &mut caches,
            &mut workspace,
            &mut Trace::off(),
        )?;
    }
    if backend.allocations() != allocations_before {
        return Err(ModelError::CheckFailed(
            "a decode step allocated a buffer".to_owned(),
        ));
    }

    // one more token than the caches have room for must be refused before
    // any layer has changed
    match model.decode_step(
        fixture.decode_token_id,
        &mut caches,
        &mut workspace,
        &mut Trace::off(),
    ) {
        Err(ModelError::CacheFull { .. }) => {}
        Err(error) => return Err(error),
        Ok(_) => {
            return Err(ModelError::CheckFailed(
                "a full KV cache accepted another token".to_owned(),
            ));
        }
    }
    if caches.iter().any(|cache| cache.len != capacity) {
        return Err(ModelError::CheckFailed(
            "a refused write still changed a KV cache".to_owned(),
        ));
    }
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

#[cfg(feature = "cuda")]
fn generate_on_gpu(
    artifacts: &Qwen3Artifacts,
    prompts: &RegressionPromptSet,
) -> Result<Vec<GenerationResult>> {
    use crate::generate::{generate, generate_uncached};
    use crate::model::Model;

    let backend = cuda_backend::Backend::new()?;
    let model = Model::load(&backend, artifacts)?;
    let max_new_tokens = prompts.max_new_tokens;
    let mut results = Vec::new();

    let mut run = |prompt: &str, prompt_ids: &[u32], eos_ids: Vec<u32>, expected: &[u32]| {
        let generated = generate(&backend, &model, prompt_ids, max_new_tokens, &eos_ids)?;
        if generated.decode_allocations > 0 {
            return Err(ModelError::CheckFailed(format!(
                "decoding {prompt:?} allocated {} buffers",
                generated.decode_allocations
            )));
        }
        let cached = generated.token_ids;
        let uncached = generate_uncached(&model, prompt_ids, max_new_tokens, &eos_ids)?;
        if cached != uncached {
            return Err(ModelError::CacheMismatch {
                prompt: prompt.to_owned(),
                cached,
                uncached,
            });
        }
        results.push(expect_tokens(prompt, &eos_ids, cached, expected)?);
        Ok(())
    };

    let eos_id = artifacts.config.eos_token_id;
    for case in &prompts.cases {
        run(
            &case.prompt,
            &case.token_ids,
            vec![eos_id],
            &case.expected_token_ids,
        )?;
    }

    // the base model rarely ends a short completion, so a token it is known
    // to generate stands in for the end-of-sequence token
    let case = &prompts.cases[0];
    run(
        &case.prompt,
        &case.token_ids,
        vec![case.expected_token_ids[1]],
        &case.expected_token_ids[..2],
    )?;

    Ok(results)
}

#[cfg(feature = "cuda")]
fn expect_tokens(
    prompt: &str,
    eos_ids: &[u32],
    token_ids: Vec<u32>,
    expected: &[u32],
) -> Result<GenerationResult> {
    if token_ids != expected {
        return Err(ModelError::GenerationMismatch {
            prompt: prompt.to_owned(),
            expected: expected.to_vec(),
            actual: token_ids,
        });
    }
    Ok(GenerationResult {
        prompt: prompt.to_owned(),
        eos_ids: eos_ids.to_vec(),
        token_ids,
    })
}

#[cfg(not(feature = "cuda"))]
fn generate_on_gpu(
    _artifacts: &Qwen3Artifacts,
    _prompts: &RegressionPromptSet,
) -> Result<Vec<GenerationResult>> {
    Err(cuda_backend::BackendError::CudaDisabled.into())
}
