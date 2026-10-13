use std::sync::Arc;
use std::time::Instant;

use cuda_core::{Device, Stream, get_device_sm_name};
use cuda_kernels::{rms_norm_bf16, rms_norm_f32, vector_add};
use cutile::half::bf16;
use cutile::prelude::*;

use crate::rms_norm::{RmsNormF32Case, reference_f32};
use crate::{
    BackendError, GpuInfo, RmsNormCaseResult, RmsNormResult, SmokeReport, VECTOR_ELEMENTS,
    VECTOR_TILE_ELEMENTS, VectorAddResult, check_rms_norm_shape,
};

const VECTOR_MAX_ABS_ERROR: f32 = 1.0e-6;
const F32_TOLERANCE: f32 = 1.0e-5;
const BF16_TOLERANCE: f32 = 1.0e-2;

pub(super) fn run_smoke(fixture: RmsNormF32Case) -> Result<SmokeReport, BackendError> {
    let visible_device_count = Device::device_count().map_err(cuda_error)?;
    if visible_device_count == 0 {
        return Err(BackendError::Cuda("no CUDA devices are visible".to_owned()));
    }

    let device = Device::new(0).map_err(cuda_error)?;
    let stream = device.new_stream().map_err(cuda_error)?;
    let name = device.name().map_err(cuda_error)?;
    let compute_capability = compute_capability(&device)?;

    let vector_add = run_vector_add(&stream)?;
    let rms_norm = run_rms_norm_suite(&stream, fixture)?;

    Ok(SmokeReport {
        gpu: GpuInfo {
            name,
            ordinal: device.ordinal(),
            compute_capability,
            visible_device_count,
        },
        cutile_version: "0.4.0",
        vector_add,
        rms_norm,
    })
}

fn run_vector_add(stream: &Arc<Stream>) -> Result<VectorAddResult, BackendError> {
    let left_host = Arc::new(
        (0_u8..17)
            .cycle()
            .map(f32::from)
            .take(VECTOR_ELEMENTS)
            .collect::<Vec<_>>(),
    );
    let right_host = Arc::new(
        (0_u8..13)
            .cycle()
            .map(f32::from)
            .take(VECTOR_ELEMENTS)
            .collect::<Vec<_>>(),
    );

    let buffer_setup_started = Instant::now();
    let left = api::copy_host_vec_to_device(&left_host)
        .sync_on(stream)
        .map_err(cutile_error)?;
    let right = api::copy_host_vec_to_device(&right_host)
        .sync_on(stream)
        .map_err(cutile_error)?;
    let mut output = api::zeros::<f32>(&[VECTOR_ELEMENTS])
        .sync_on(stream)
        .map_err(cutile_error)?;
    let buffer_setup_ms = elapsed_ms(buffer_setup_started);

    let first_run_started = Instant::now();
    let _ = vector_add(
        (&mut output).partition([VECTOR_TILE_ELEMENTS]),
        &left,
        &right,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;
    let first_run_ms = elapsed_ms(first_run_started);

    let warm_run_started = Instant::now();
    let _ = vector_add(
        (&mut output).partition([VECTOR_TILE_ELEMENTS]),
        &left,
        &right,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;
    let warm_run_ms = elapsed_ms(warm_run_started);

    let download_started = Instant::now();
    let output_host = output.to_host_vec().sync_on(stream).map_err(cutile_error)?;
    let download_ms = elapsed_ms(download_started);

    let expected = left_host
        .iter()
        .zip(right_host.iter())
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    let max_abs_error = verify("vector_add", &output_host, &expected, VECTOR_MAX_ABS_ERROR)?;

    Ok(VectorAddResult {
        elements: VECTOR_ELEMENTS,
        tile_elements: VECTOR_TILE_ELEMENTS,
        buffer_setup_ms,
        first_run_ms,
        warm_run_ms,
        download_ms,
        max_abs_error,
        passed: true,
    })
}

fn run_rms_norm_suite(
    stream: &Arc<Stream>,
    fixture: RmsNormF32Case,
) -> Result<RmsNormResult, BackendError> {
    let mut cases = Vec::with_capacity(3);

    cases.push(run_rms_norm_f32(
        stream,
        "qwen3_fixture_f32",
        "golden_fixture",
        fixture,
    )?);

    let rows = 3;
    let hidden_size = 128;
    let epsilon = 1.0e-6;
    let input = synthetic_input(rows * hidden_size);
    let weight = synthetic_weight(hidden_size);
    let expected = reference_f32(&input, &weight, rows, hidden_size, epsilon);
    cases.push(run_rms_norm_f32(
        stream,
        "synthetic_128_f32",
        "rust_reference",
        RmsNormF32Case {
            rows,
            hidden_size,
            input,
            weight,
            epsilon,
            expected,
        },
    )?);

    cases.push(run_rms_norm_bf16(stream)?);

    Ok(RmsNormResult { cases })
}

fn run_rms_norm_f32(
    stream: &Arc<Stream>,
    name: &'static str,
    source: &'static str,
    case: RmsNormF32Case,
) -> Result<RmsNormCaseResult, BackendError> {
    let RmsNormF32Case {
        rows,
        hidden_size,
        input,
        weight,
        epsilon,
        expected,
    } = case;
    check_rms_norm_shape(rows, hidden_size)?;
    let inverse_hidden_size = inverse_hidden_size(hidden_size);
    let input_host = Arc::new(input);
    let weight_host = Arc::new(weight);

    let input = api::copy_host_vec_to_device(&input_host)
        .sync_on(stream)
        .map_err(cutile_error)?
        .reshape(&[rows, hidden_size])
        .map_err(cutile_error)?;
    let weight = api::copy_host_vec_to_device(&weight_host)
        .sync_on(stream)
        .map_err(cutile_error)?;
    let mut output = api::zeros::<f32>(&[rows, hidden_size])
        .sync_on(stream)
        .map_err(cutile_error)?;

    let _ = rms_norm_f32(
        (&mut output).partition([1, hidden_size]),
        &input,
        &weight,
        inverse_hidden_size,
        epsilon,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;

    let actual = output.to_host_vec().sync_on(stream).map_err(cutile_error)?;
    let max_abs_error = verify(name, &actual, &expected, F32_TOLERANCE)?;

    Ok(RmsNormCaseResult {
        name,
        source,
        dtype: "float32",
        rows,
        hidden_size,
        epsilon,
        max_abs_error,
        tolerance: F32_TOLERANCE,
        passed: true,
    })
}

fn run_rms_norm_bf16(stream: &Arc<Stream>) -> Result<RmsNormCaseResult, BackendError> {
    let name = "synthetic_512_bfloat16";
    let rows = 2;
    let hidden_size = 512;
    let epsilon = 1.0e-6;
    check_rms_norm_shape(rows, hidden_size)?;

    let input_host = Arc::new(
        synthetic_input(rows * hidden_size)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    );
    let weight_host = Arc::new(
        synthetic_weight(hidden_size)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    );
    let expected = reference_bf16(&input_host, &weight_host, rows, hidden_size, epsilon);

    let input = api::copy_host_vec_to_device(&input_host)
        .sync_on(stream)
        .map_err(cutile_error)?
        .reshape(&[rows, hidden_size])
        .map_err(cutile_error)?;
    let weight = api::copy_host_vec_to_device(&weight_host)
        .sync_on(stream)
        .map_err(cutile_error)?;
    let mut output = api::zeros::<bf16>(&[rows, hidden_size])
        .sync_on(stream)
        .map_err(cutile_error)?;

    let _ = rms_norm_bf16(
        (&mut output).partition([1, hidden_size]),
        &input,
        &weight,
        inverse_hidden_size(hidden_size),
        epsilon,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;

    let actual = output.to_host_vec().sync_on(stream).map_err(cutile_error)?;
    let actual_f32 = actual
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let expected_f32 = expected
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let max_abs_error = verify(name, &actual_f32, &expected_f32, BF16_TOLERANCE)?;

    Ok(RmsNormCaseResult {
        name,
        source: "rust_reference",
        dtype: "bfloat16",
        rows,
        hidden_size,
        epsilon,
        max_abs_error,
        tolerance: BF16_TOLERANCE,
        passed: true,
    })
}

fn reference_bf16(
    input: &[bf16],
    weight: &[bf16],
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
) -> Vec<bf16> {
    let hidden_size_f32 =
        f32::from(u16::try_from(hidden_size).expect("validated RMSNorm hidden size fits in u16"));
    let mut output = Vec::with_capacity(input.len());

    for row in input.chunks_exact(hidden_size).take(rows) {
        let mean_square = row
            .iter()
            .map(|value| {
                let value = value.to_f32();
                value * value
            })
            .sum::<f32>()
            / hidden_size_f32;
        let inverse_rms = (mean_square + epsilon).sqrt().recip();
        output.extend(row.iter().zip(weight).map(|(value, scale)| {
            let normalized = bf16::from_f32(value.to_f32() * inverse_rms);
            bf16::from_f32(normalized.to_f32() * scale.to_f32())
        }));
    }

    output
}

fn synthetic_input(elements: usize) -> Vec<f32> {
    (-16_i16..=16)
        .cycle()
        .map(|value| f32::from(value) / 8.0)
        .take(elements)
        .collect()
}

fn synthetic_weight(hidden_size: usize) -> Vec<f32> {
    (0_i16..11)
        .cycle()
        .map(|value| 0.9 + f32::from(value) / 100.0)
        .take(hidden_size)
        .collect()
}

fn inverse_hidden_size(hidden_size: usize) -> f32 {
    let hidden_size =
        u16::try_from(hidden_size).expect("validated RMSNorm hidden size fits in u16");
    f32::from(hidden_size).recip()
}

fn verify(
    case: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<f32, BackendError> {
    if actual.len() != expected.len() {
        return Err(BackendError::OutputLength {
            case: case.to_owned(),
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    let mut max_abs_error = 0.0_f32;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        if !error.is_finite() {
            max_abs_error = f32::INFINITY;
            break;
        }
        max_abs_error = max_abs_error.max(error);
    }

    if max_abs_error > tolerance {
        return Err(BackendError::VerificationFailed {
            case: case.to_owned(),
            max_abs_error,
            tolerance,
        });
    }
    Ok(max_abs_error)
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

#[allow(unsafe_code)]
fn compute_capability(device: &Device) -> Result<String, BackendError> {
    // SAFETY: Device::new created this handle and keeps it alive for the query.
    unsafe { get_device_sm_name(device.cu_device()) }.map_err(cuda_error)
}
