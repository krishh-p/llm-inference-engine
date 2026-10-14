use std::path::Path;

use engine_core::{FixtureManifest, SafeTensorCatalog, Tolerance};

use crate::weights::{LayerWeights, load_tensor};
use crate::{Qwen3Artifacts, Qwen3Config, Result};

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

    let mut trace = Trace::default();
    let x = backend.upload(&prefill.input, &[prefill.rows, config.hidden_size])?;
    let (_, cache) = block.forward(&x, prefill.rows, None, &mut trace)?;
    compare(prefill, &trace, &mut results)?;

    // decode attends to the cache that prefill just built
    let mut trace = Trace::default();
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
