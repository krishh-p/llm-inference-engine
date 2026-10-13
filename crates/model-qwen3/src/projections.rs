use std::path::Path;

use engine_core::{FixtureManifest, SafeTensorCatalog, Tolerance};

use crate::{ModelError, Qwen3Artifacts, Qwen3Config, Result};

// The golden fixtures trace layer 0, once during prefill and once during decode.
const WEIGHT_PREFIX: &str = "model.layers.0.self_attn.";
const FIXTURES: [(&str, &str); 2] = [
    ("prefill", "block_0.safetensors"),
    ("decode", "decode_block_0.safetensors"),
];

/// How closely one GPU projection matched its fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionResult {
    pub name: String,
    pub shape: [usize; 3],
    pub max_abs_error: f32,
    pub tolerance: Tolerance,
}

/// One layer 0 projection weight with the fixtures that exercise it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct Projection {
    /// `q`, `k` or `v`, which is also the name of the fixture tensor it produces.
    pub name: &'static str,
    pub heads: usize,
    /// Row-major `[heads * head_dim, hidden_size]`, widened to float32.
    pub weight: Vec<f32>,
    pub cases: Vec<ProjectionCase>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct ProjectionCase {
    pub name: &'static str,
    pub rows: usize,
    /// The normalized layer input, `[rows, hidden_size]`.
    pub input: Vec<f32>,
    /// The projected rows after the head split, `[heads, rows, head_dim]`.
    pub expected: Vec<f32>,
    pub tolerance: Tolerance,
}

/// Runs layer 0's Q, K and V projections on the GPU with the real weights and
/// compares each one with the golden fixtures.
///
/// # Errors
///
/// Returns an error when a fixture or weight cannot be loaded, the GPU is
/// unavailable, or a projection is outside its fixture tolerance.
pub fn verify_projections(
    artifacts: &Qwen3Artifacts,
    fixture_dir: impl AsRef<Path>,
) -> Result<Vec<ProjectionResult>> {
    let fixture_dir = fixture_dir.as_ref();
    let config = &artifacts.config;
    let manifest = FixtureManifest::load(fixture_dir.join("manifest.json"))?;
    manifest.validate_files(fixture_dir)?;

    let projections = [
        ("q", config.num_attention_heads),
        ("k", config.num_key_value_heads),
        ("v", config.num_key_value_heads),
    ]
    .into_iter()
    .map(|(name, heads)| load_projection(artifacts, &manifest, fixture_dir, name, heads))
    .collect::<Result<Vec<_>>>()?;

    run_on_gpu(config, &projections)
}

fn load_projection(
    artifacts: &Qwen3Artifacts,
    manifest: &FixtureManifest,
    fixture_dir: &Path,
    name: &'static str,
    heads: usize,
) -> Result<Projection> {
    let config = &artifacts.config;
    let weight = load_tensor(
        &artifacts.weights,
        &format!("{WEIGHT_PREFIX}{name}_proj.weight"),
        &[heads * config.head_dim, config.hidden_size],
    )?;

    let mut cases = Vec::with_capacity(FIXTURES.len());
    for (case, file) in FIXTURES {
        let catalog = SafeTensorCatalog::open_files([fixture_dir.join(file)])?;
        let rows = catalog
            .get("input_norm")
            .and_then(|tensor| tensor.layout.shape.first().copied())
            .ok_or_else(|| ModelError::MissingTensor("input_norm".to_owned()))?;

        cases.push(ProjectionCase {
            name: case,
            rows,
            input: load_tensor(&catalog, "input_norm", &[rows, config.hidden_size])?,
            expected: load_tensor(&catalog, name, &[heads, rows, config.head_dim])?,
            tolerance: manifest.tolerance(file, "projection")?,
        });
    }

    Ok(Projection {
        name,
        heads,
        weight,
        cases,
    })
}

/// Loads a tensor as float32 after checking its shape.
fn load_tensor(catalog: &SafeTensorCatalog, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let tensor = catalog.load(name)?;
    if tensor.descriptor.layout.shape != shape {
        return Err(ModelError::ShapeMismatch {
            name: name.to_owned(),
            expected: shape.to_vec(),
            actual: tensor.descriptor.layout.shape,
        });
    }
    Ok(tensor.widen_to_f32_vec()?)
}

/// Rearranges `[rows, heads * head_dim]` into `[heads, rows, head_dim]`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn split_heads(values: &[f32], rows: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(values.len());

    for head in 0..heads {
        for row in 0..rows {
            let start = (row * heads + head) * head_dim;
            output.extend_from_slice(&values[start..start + head_dim]);
        }
    }

    output
}

/// Returns the largest absolute error between `actual` and `expected`.
///
/// # Errors
///
/// Returns an error when any value is outside the tolerance.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: Tolerance,
) -> Result<f32> {
    let mut max_abs_error = 0.0_f32;
    let mut within_tolerance = actual.len() == expected.len();

    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        if !error.is_finite() {
            max_abs_error = f32::INFINITY;
            break;
        }
        max_abs_error = max_abs_error.max(error);
        within_tolerance &= tolerance.allows(actual, expected);
    }

    if !within_tolerance || !max_abs_error.is_finite() {
        return Err(ModelError::FixtureMismatch {
            name: name.to_owned(),
            max_abs_error,
            atol: tolerance.atol,
            rtol: tolerance.rtol,
        });
    }
    Ok(max_abs_error)
}

#[cfg(feature = "cuda")]
fn run_on_gpu(config: &Qwen3Config, projections: &[Projection]) -> Result<Vec<ProjectionResult>> {
    crate::gpu::run_projections(config, projections)
}

#[cfg(not(feature = "cuda"))]
fn run_on_gpu(_config: &Qwen3Config, _projections: &[Projection]) -> Result<Vec<ProjectionResult>> {
    Err(cuda_backend::BackendError::CudaDisabled.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: Tolerance = Tolerance {
        atol: 0.5,
        rtol: 0.0,
    };

    #[test]
    fn split_heads_groups_rows_by_head() {
        // two rows of two heads, where each value is row, head, element
        let values = [
            111.0, 112.0, 121.0, 122.0, // row 1: head 1, head 2
            211.0, 212.0, 221.0, 222.0, // row 2: head 1, head 2
        ];

        assert_eq!(
            split_heads(&values, 2, 2, 2),
            vec![
                111.0, 112.0, 211.0, 212.0, // head 1: row 1, row 2
                121.0, 122.0, 221.0, 222.0, // head 2: row 1, row 2
            ]
        );
    }

    #[test]
    fn compare_returns_the_largest_error() {
        let max_abs_error = compare("case", &[1.0, 2.25, 3.0], &[1.0, 2.0, 3.5], TOLERANCE);

        assert_eq!(max_abs_error.unwrap(), 0.5);
    }

    #[test]
    fn compare_rejects_a_value_outside_the_tolerance() {
        let error = compare("prefill_q", &[1.0, 3.0], &[1.0, 2.0], TOLERANCE).unwrap_err();

        assert_eq!(
            error.to_string(),
            "prefill_q differs from its fixture by up to 1, allowed 0.5 + 0 * |expected|"
        );
    }

    #[test]
    fn compare_rejects_nan() {
        assert!(matches!(
            compare("case", &[f32::NAN], &[1.0], TOLERANCE),
            Err(ModelError::FixtureMismatch { .. })
        ));
    }
}
