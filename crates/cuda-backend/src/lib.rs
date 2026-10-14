use std::path::Path;

use engine_core::Tolerance;
use serde::Serialize;
use thiserror::Error;

mod matmul;
mod rms_norm;
mod rope;
mod shapes;

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod cublas;
#[cfg(feature = "cuda")]
mod gpu;

#[cfg(feature = "cuda")]
pub use backend::{Backend, reshape};
#[cfg(feature = "cuda")]
pub use cutile::tensor::Tensor;
pub use rope::rope_tables;

const VECTOR_ELEMENTS: usize = 4096;
const VECTOR_TILE_ELEMENTS: usize = 256;
const RMS_NORM_HIDDEN_SIZE_MULTIPLE: usize = 32;
const MAX_RMS_NORM_HIDDEN_SIZE: usize = 8192;

/// GPU information captured by the smoke test.
#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub ordinal: usize,
    pub compute_capability: String,
    pub visible_device_count: i32,
}

/// Timing and correctness data for vector add.
#[derive(Debug, Serialize)]
pub struct VectorAddResult {
    pub elements: usize,
    pub tile_elements: usize,
    pub buffer_setup_ms: f64,
    pub first_run_ms: f64,
    pub warm_run_ms: f64,
    pub download_ms: f64,
    pub max_abs_error: f32,
    pub passed: bool,
}

/// Correctness data for one `RMSNorm` shape and dtype.
#[derive(Debug, Serialize)]
pub struct RmsNormCaseResult {
    pub name: &'static str,
    pub source: &'static str,
    pub dtype: &'static str,
    pub rows: usize,
    pub hidden_size: usize,
    pub epsilon: f32,
    pub max_abs_error: f32,
    pub tolerance: f32,
    pub benchmark: KernelBenchmark,
    pub passed: bool,
}

/// Repeated warm-kernel timing for one case.
#[derive(Debug, Serialize)]
pub struct KernelBenchmark {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub min_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub effective_bandwidth_gbps: f64,
    pub launches_per_iteration: usize,
}

/// A residual add followed by `RMSNorm`, timed as two kernels and as one
/// fused kernel on the same inputs.
#[derive(Debug, Serialize)]
pub struct FusedNormCaseResult {
    pub rows: usize,
    pub hidden_size: usize,
    pub max_abs_difference: f32,
    pub tolerance: f32,
    pub unfused: KernelBenchmark,
    pub fused: KernelBenchmark,
    pub speedup: f64,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct FusedNormResult {
    pub cases: Vec<FusedNormCaseResult>,
}

#[derive(Debug, Serialize)]
pub struct RmsNormResult {
    pub cases: Vec<RmsNormCaseResult>,
}

/// Correctness data for one cuBLAS matrix multiplication.
#[derive(Debug, Serialize)]
pub struct MatmulCaseResult {
    pub name: &'static str,
    pub left_shape: [usize; 2],
    pub right_shape: [usize; 2],
    pub right_transposed: bool,
    pub max_abs_error: f32,
    pub tolerance: f32,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct MatmulResult {
    pub cases: Vec<MatmulCaseResult>,
}

/// How closely the `RoPE` tables generated in Rust match the reference tables.
#[derive(Debug, Serialize)]
pub struct RopeTableResult {
    pub positions: Vec<u32>,
    pub cos_max_abs_error: f32,
    pub sin_max_abs_error: f32,
    pub tolerance: Tolerance,
    pub passed: bool,
}

/// Correctness data for one tensor rotated by the `RoPE` kernel.
#[derive(Debug, Serialize)]
pub struct RopeCaseResult {
    pub name: &'static str,
    pub heads: usize,
    pub rows: usize,
    pub head_dim: usize,
    pub max_abs_error: f32,
    pub tolerance: Tolerance,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct RopeResult {
    pub tables: RopeTableResult,
    pub cases: Vec<RopeCaseResult>,
}

#[derive(Debug, Serialize)]
pub struct SmokeReport {
    pub gpu: GpuInfo,
    pub cutile_version: &'static str,
    pub vector_add: VectorAddResult,
    pub rms_norm: RmsNormResult,
    pub fused_norm: FusedNormResult,
    pub rope: RopeResult,
    pub matmul: MatmulResult,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("element count must be a non-zero multiple of {VECTOR_TILE_ELEMENTS}, got {0}")]
    InvalidElementCount(usize),

    #[error("RMSNorm needs at least one row and one hidden element, got [{rows}, {hidden_size}]")]
    InvalidRmsNormShape { rows: usize, hidden_size: usize },

    #[error(
        "RMSNorm currently supports hidden sizes divisible by {RMS_NORM_HIDDEN_SIZE_MULTIPLE} up to {MAX_RMS_NORM_HIDDEN_SIZE}, got {0}"
    )]
    UnsupportedHiddenSize(usize),

    #[error(
        "matmul shapes do not agree: left {left:?}, right {right:?}, output {output:?}, right transposed: {right_transposed}"
    )]
    MatmulShape {
        left: Vec<i32>,
        right: Vec<i32>,
        output: Vec<i32>,
        right_transposed: bool,
    },

    #[error("{operation} {name} has shape {actual:?}, expected {expected:?}")]
    Shape {
        operation: &'static str,
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<i32>,
    },

    #[error("{operation} {name} has shape {actual:?}, expected {expected} dimensions")]
    Rank {
        operation: &'static str,
        name: &'static str,
        expected: usize,
        actual: Vec<i32>,
    },

    #[error("{operation} {name} needs {rows} rows from row {first}, but has {available} rows")]
    RowRange {
        operation: &'static str,
        name: &'static str,
        first: usize,
        rows: usize,
        available: usize,
    },

    #[error("{0} tensor must be contiguous and row-major")]
    NonContiguous(&'static str),

    #[error("{name} tensor is on CUDA device {actual}, but the backend uses device {expected}")]
    WrongDevice {
        name: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("this build has no CUDA support; rebuild with the cuda feature")]
    CudaDisabled,

    #[error("CUDA setup failed: {0}")]
    Cuda(String),

    #[error("cuTile failed: {0}")]
    Cutile(String),

    #[error("cuBLAS {operation} failed with {name} ({code})")]
    Cublas {
        operation: &'static str,
        name: &'static str,
        code: i32,
    },

    #[error("invalid RMSNorm fixture: {0}")]
    RmsNormFixture(String),

    #[error("invalid RoPE fixture: {0}")]
    RopeFixture(String),

    #[error("{case} returned {actual} values, expected {expected}")]
    OutputLength {
        case: String,
        expected: usize,
        actual: usize,
    },

    #[error("{case} max error {max_abs_error} exceeds {tolerance}")]
    VerificationFailed {
        case: String,
        max_abs_error: f32,
        tolerance: f32,
    },

    #[error(transparent)]
    Engine(#[from] engine_core::EngineError),
}

/// Finds the first GPU and runs the verified smoke kernels.
///
/// # Errors
///
/// Returns an error when a fixture is invalid, CUDA is unavailable or a
/// kernel fails verification.
pub fn run_smoke(fixture_dir: impl AsRef<Path>) -> Result<SmokeReport, BackendError> {
    let fixture_dir = fixture_dir.as_ref();
    check_element_count(VECTOR_ELEMENTS)?;
    let rms_norm = rms_norm::load_fixtures(&fixture_dir.join("rms_norm.safetensors"))?;
    let rope = rope::load_fixtures(fixture_dir)?;
    run_smoke_on_gpu(rms_norm, &rope)
}

fn check_element_count(elements: usize) -> Result<(), BackendError> {
    if elements == 0 || !elements.is_multiple_of(VECTOR_TILE_ELEMENTS) {
        return Err(BackendError::InvalidElementCount(elements));
    }
    Ok(())
}

fn check_rms_norm_shape(rows: usize, hidden_size: usize) -> Result<(), BackendError> {
    if rows == 0 || hidden_size == 0 {
        return Err(BackendError::InvalidRmsNormShape { rows, hidden_size });
    }
    // This kernel maps one complete row to a tile. Widths 128, 2048 and 2560
    // have been run on the GPU. Other multiples of 32 are accepted but untested.
    if !hidden_size.is_multiple_of(RMS_NORM_HIDDEN_SIZE_MULTIPLE)
        || hidden_size > MAX_RMS_NORM_HIDDEN_SIZE
    {
        return Err(BackendError::UnsupportedHiddenSize(hidden_size));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_smoke_on_gpu(
    rms_norm: rms_norm::RmsNormFixtures,
    rope: &rope::RopeFixtures,
) -> Result<SmokeReport, BackendError> {
    gpu::run_smoke(rms_norm, rope)
}

#[cfg(not(feature = "cuda"))]
fn run_smoke_on_gpu(
    _rms_norm: rms_norm::RmsNormFixtures,
    _rope: &rope::RopeFixtures,
) -> Result<SmokeReport, BackendError> {
    Err(BackendError::CudaDisabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_vector_input() {
        assert!(matches!(
            check_element_count(0),
            Err(BackendError::InvalidElementCount(0))
        ));
    }

    #[test]
    fn rejects_partial_vector_tile() {
        assert!(matches!(
            check_element_count(VECTOR_TILE_ELEMENTS + 1),
            Err(BackendError::InvalidElementCount(_))
        ));
    }

    #[test]
    fn accepts_qwen3_hidden_size() {
        assert!(check_rms_norm_shape(5, 2048).is_ok());
    }

    #[test]
    fn accepts_qwen3_4b_hidden_size() {
        assert!(check_rms_norm_shape(2, 2560).is_ok());
    }

    #[test]
    fn rejects_odd_hidden_size() {
        assert!(matches!(
            check_rms_norm_shape(1, 257),
            Err(BackendError::UnsupportedHiddenSize(257))
        ));
    }
}
