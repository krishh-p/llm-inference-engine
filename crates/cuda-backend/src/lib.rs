use std::path::Path;

use serde::Serialize;
use thiserror::Error;

mod rms_norm;

#[cfg(target_os = "linux")]
mod gpu;

const VECTOR_ELEMENTS: usize = 4096;
const VECTOR_TILE_ELEMENTS: usize = 256;
// Only power-of-two widths up to 2048 have been run on the GPU so far. This is
// the validated range, not a known cuTile limit.
const MAX_RMS_NORM_HIDDEN_SIZE: usize = 2048;

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
    pub passed: bool,
}

/// `RMSNorm` correctness results.
#[derive(Debug, Serialize)]
pub struct RmsNormResult {
    pub cases: Vec<RmsNormCaseResult>,
}

/// Complete output from the Rust CUDA smoke test.
#[derive(Debug, Serialize)]
pub struct SmokeReport {
    pub gpu: GpuInfo,
    pub cutile_version: &'static str,
    pub vector_add: VectorAddResult,
    pub rms_norm: RmsNormResult,
}

/// Errors raised while checking the CUDA runtime or running a kernel.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("element count must be a non-zero multiple of {VECTOR_TILE_ELEMENTS}, got {0}")]
    InvalidElementCount(usize),

    #[error("RMSNorm needs at least one row and one hidden element, got [{rows}, {hidden_size}]")]
    InvalidRmsNormShape { rows: usize, hidden_size: usize },

    #[error("RMSNorm supports power-of-two hidden sizes up to {MAX_RMS_NORM_HIDDEN_SIZE}, got {0}")]
    UnsupportedHiddenSize(usize),

    #[error("the CUDA backend requires Linux")]
    UnsupportedPlatform,

    #[error("CUDA setup failed: {0}")]
    Cuda(String),

    #[error("cuTile failed: {0}")]
    Cutile(String),

    #[error("invalid RMSNorm fixture: {0}")]
    RmsNormFixture(String),

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
/// Returns an error when the fixture is invalid, CUDA is unavailable or a
/// kernel fails verification.
pub fn run_smoke(rms_norm_fixture: impl AsRef<Path>) -> Result<SmokeReport, BackendError> {
    check_element_count(VECTOR_ELEMENTS)?;
    let fixture = rms_norm::load_fixture(rms_norm_fixture.as_ref())?;
    run_smoke_on_gpu(fixture)
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
    if !hidden_size.is_power_of_two() || hidden_size > MAX_RMS_NORM_HIDDEN_SIZE {
        return Err(BackendError::UnsupportedHiddenSize(hidden_size));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn run_smoke_on_gpu(fixture: rms_norm::RmsNormF32Case) -> Result<SmokeReport, BackendError> {
    gpu::run_smoke(fixture)
}

#[cfg(not(target_os = "linux"))]
fn run_smoke_on_gpu(_fixture: rms_norm::RmsNormF32Case) -> Result<SmokeReport, BackendError> {
    Err(BackendError::UnsupportedPlatform)
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
    fn rejects_odd_hidden_size() {
        assert!(matches!(
            check_rms_norm_shape(1, 257),
            Err(BackendError::UnsupportedHiddenSize(257))
        ));
    }
}
