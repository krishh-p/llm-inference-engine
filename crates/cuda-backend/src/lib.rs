use serde::Serialize;
use thiserror::Error;

const VECTOR_ELEMENTS: usize = 4096;
const TILE_ELEMENTS: usize = 256;
#[cfg(target_os = "linux")]
const MAX_ABS_ERROR: f32 = 1.0e-6;

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

/// Complete output from the Rust CUDA smoke test.
#[derive(Debug, Serialize)]
pub struct SmokeReport {
    pub gpu: GpuInfo,
    pub cutile_version: &'static str,
    pub vector_add: VectorAddResult,
}

/// Errors raised while checking the CUDA runtime or running the smoke kernel.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("element count must be a non-zero multiple of {TILE_ELEMENTS}, got {0}")]
    InvalidElementCount(usize),

    #[error("the CUDA backend requires Linux")]
    UnsupportedPlatform,

    #[error("CUDA setup failed: {0}")]
    Cuda(String),

    #[error("cuTile failed: {0}")]
    Cutile(String),

    #[error("vector add max error {max_abs_error} exceeds {tolerance}")]
    VerificationFailed { max_abs_error: f32, tolerance: f32 },
}

/// Finds the first GPU and runs the verified vector-add smoke kernel.
///
/// # Errors
///
/// Returns an error when CUDA is unavailable or the kernel fails verification.
pub fn run_smoke() -> Result<SmokeReport, BackendError> {
    check_element_count(VECTOR_ELEMENTS)?;
    run_smoke_on_gpu()
}

fn check_element_count(elements: usize) -> Result<(), BackendError> {
    if elements == 0 || !elements.is_multiple_of(TILE_ELEMENTS) {
        return Err(BackendError::InvalidElementCount(elements));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn run_smoke_on_gpu() -> Result<SmokeReport, BackendError> {
    gpu::run_smoke()
}

#[cfg(not(target_os = "linux"))]
fn run_smoke_on_gpu() -> Result<SmokeReport, BackendError> {
    Err(BackendError::UnsupportedPlatform)
}

#[cfg(target_os = "linux")]
mod gpu {
    use std::sync::Arc;
    use std::time::Instant;

    use cuda_core::CudaContext;
    use cuda_kernels::vector_add;
    use cutile::prelude::*;

    use super::{
        BackendError, GpuInfo, MAX_ABS_ERROR, SmokeReport, TILE_ELEMENTS, VECTOR_ELEMENTS,
        VectorAddResult,
    };

    pub(super) fn run_smoke() -> Result<SmokeReport, BackendError> {
        let visible_device_count = Device::device_count().map_err(cuda_error)?;
        if visible_device_count == 0 {
            return Err(BackendError::Cuda("no CUDA devices are visible".to_owned()));
        }

        let context = CudaContext::new(0).map_err(cuda_error)?;
        let name = context.device_name().map_err(cuda_error)?;
        let (major, minor) = context.compute_capability().map_err(cuda_error)?;
        let compute_capability = format!("sm_{major}{minor}");

        let result = run_vector_add(VECTOR_ELEMENTS)?;

        Ok(SmokeReport {
            gpu: GpuInfo {
                name,
                ordinal: context.ordinal(),
                compute_capability,
                visible_device_count,
            },
            cutile_version: "0.4.0",
            vector_add: result,
        })
    }

    /// Runs vector add once through the JIT path and once through the warm path.
    fn run_vector_add(elements: usize) -> Result<VectorAddResult, BackendError> {
        let left_host = Arc::new(
            (0_u8..17)
                .cycle()
                .map(f32::from)
                .take(elements)
                .collect::<Vec<_>>(),
        );
        let right_host = Arc::new(
            (0_u8..13)
                .cycle()
                .map(f32::from)
                .take(elements)
                .collect::<Vec<_>>(),
        );

        let device = Device::new(0).map_err(cutile_error)?;
        let stream = device.new_stream().map_err(cutile_error)?;

        let buffer_setup_started = Instant::now();
        let left = api::copy_host_vec_to_device(&left_host)
            .sync_on(&stream)
            .map_err(cutile_error)?;
        let right = api::copy_host_vec_to_device(&right_host)
            .sync_on(&stream)
            .map_err(cutile_error)?;
        let mut output = api::zeros::<f32>(&[elements])
            .sync_on(&stream)
            .map_err(cutile_error)?;
        let buffer_setup_ms = elapsed_ms(buffer_setup_started);

        let first_run_started = Instant::now();
        let _ = vector_add((&mut output).partition([TILE_ELEMENTS]), &left, &right)
            .sync_on(&stream)
            .map_err(cutile_error)?;
        let first_run_ms = elapsed_ms(first_run_started);

        let warm_run_started = Instant::now();
        let _ = vector_add((&mut output).partition([TILE_ELEMENTS]), &left, &right)
            .sync_on(&stream)
            .map_err(cutile_error)?;
        let warm_run_ms = elapsed_ms(warm_run_started);

        let download_started = Instant::now();
        let output_host = output
            .to_host_vec()
            .sync_on(&stream)
            .map_err(cutile_error)?;
        let download_ms = elapsed_ms(download_started);

        let max_abs_error = output_host
            .iter()
            .zip(left_host.iter().zip(right_host.iter()))
            .map(|(actual, (left, right))| (actual - (left + right)).abs())
            .fold(0.0_f32, f32::max);

        if max_abs_error > MAX_ABS_ERROR {
            return Err(BackendError::VerificationFailed {
                max_abs_error,
                tolerance: MAX_ABS_ERROR,
            });
        }

        Ok(VectorAddResult {
            elements,
            tile_elements: TILE_ELEMENTS,
            buffer_setup_ms,
            first_run_ms,
            warm_run_ms,
            download_ms,
            max_abs_error,
            passed: true,
        })
    }

    fn elapsed_ms(started: Instant) -> f64 {
        started.elapsed().as_secs_f64() * 1_000.0
    }

    fn cuda_error(error: impl std::fmt::Display) -> BackendError {
        BackendError::Cuda(error.to_string())
    }

    fn cutile_error(error: impl std::fmt::Display) -> BackendError {
        BackendError::Cutile(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_input() {
        assert!(matches!(
            check_element_count(0),
            Err(BackendError::InvalidElementCount(0))
        ));
    }

    #[test]
    fn rejects_partial_tile() {
        assert!(matches!(
            check_element_count(TILE_ELEMENTS + 1),
            Err(BackendError::InvalidElementCount(_))
        ));
    }
}
