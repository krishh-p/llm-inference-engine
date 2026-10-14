use std::sync::Arc;
use std::time::Instant;

use cuda_core::{Device, Stream, get_device_sm_name};
use cuda_kernels::{rms_norm_bf16, vector_add};
use cutile::half::bf16;
use cutile::prelude::*;

use crate::backend::Backend;
use crate::matmul::reference;
use crate::rms_norm::{RmsNormBf16Case, RmsNormF32Case, RmsNormFixtures, inverse_hidden_size};
use crate::rope::{RopeCase, RopeFixtures, check_tables};
use crate::shapes::dims;
use crate::{
    BackendError, GpuInfo, KernelBenchmark, MatmulCaseResult, MatmulResult, RmsNormCaseResult,
    RmsNormResult, RopeCaseResult, RopeResult, SmokeReport, VECTOR_ELEMENTS, VECTOR_TILE_ELEMENTS,
    VectorAddResult, check_rms_norm_shape,
};

const VECTOR_MAX_ABS_ERROR: f32 = 1.0e-6;
// The matmul inputs are small whole numbers, so every product is exact in float32.
const MATMUL_MAX_ABS_ERROR: f32 = 1.0e-6;
// A 16 x 16 matrix is exactly one vector-add tile.
const KERNEL_MATMUL_SIZE: usize = 16;
const F32_TOLERANCE: f32 = 1.0e-5;
const BF16_TOLERANCE: f32 = 1.0e-2;
const BENCHMARK_WARMUP_ITERATIONS: usize = 10;
const BENCHMARK_ITERATIONS: usize = 100;

struct RmsNormParams {
    rows: usize,
    hidden_size: usize,
    tile_hidden_size: usize,
    inverse_hidden_size: f32,
    epsilon: f32,
}

impl RmsNormParams {
    fn new(rows: usize, hidden_size: usize, epsilon: f32) -> Self {
        Self {
            rows,
            hidden_size,
            // The tile width is rounded up to a power of two. The padding adds
            // nothing to the sum, and the mean still divides by the real width.
            tile_hidden_size: hidden_size.next_power_of_two(),
            inverse_hidden_size: inverse_hidden_size(hidden_size),
            epsilon,
        }
    }
}

pub(super) fn run_smoke(
    rms_norm_fixtures: RmsNormFixtures,
    rope_fixtures: &RopeFixtures,
) -> Result<SmokeReport, BackendError> {
    let visible_device_count = Device::device_count().map_err(cuda_error)?;
    if visible_device_count == 0 {
        return Err(BackendError::Cuda("no CUDA devices are visible".to_owned()));
    }

    let backend = Backend::new()?;
    let name = backend.device.name().map_err(cuda_error)?;
    let compute_capability = compute_capability(&backend.device)?;

    let vector_add = run_vector_add(&backend.stream)?;
    let rms_norm = run_rms_norm_suite(&backend, rms_norm_fixtures)?;
    let rope = run_rope_suite(&backend, rope_fixtures)?;
    let matmul = run_matmul_suite(&backend)?;

    Ok(SmokeReport {
        gpu: GpuInfo {
            name,
            ordinal: backend.device.ordinal(),
            compute_capability,
            visible_device_count,
        },
        cutile_version: "0.4.0",
        vector_add,
        rms_norm,
        rope,
        matmul,
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
    backend: &Backend,
    fixtures: RmsNormFixtures,
) -> Result<RmsNormResult, BackendError> {
    let case_count = fixtures.f32_cases.len() + fixtures.bf16_cases.len();
    let mut cases = Vec::with_capacity(case_count);

    for case in fixtures.f32_cases {
        cases.push(run_rms_norm_f32(backend, case, fixtures.epsilon)?);
    }
    for case in fixtures.bf16_cases {
        cases.push(run_rms_norm_bf16(&backend.stream, case, fixtures.epsilon)?);
    }

    Ok(RmsNormResult { cases })
}

fn run_rms_norm_f32(
    backend: &Backend,
    case: RmsNormF32Case,
    epsilon: f32,
) -> Result<RmsNormCaseResult, BackendError> {
    let RmsNormF32Case {
        name,
        rows,
        hidden_size,
        input,
        weight,
        expected,
    } = case;
    check_rms_norm_shape(rows, hidden_size)?;

    let input = backend.upload(&input, &[rows, hidden_size])?;
    let weight = backend.upload(&weight, &[hidden_size])?;
    let mut output = backend.zeros(&[rows, hidden_size])?;

    backend.rms_norm(&mut output, &input, &weight, epsilon)?;
    let benchmark = benchmark_rms_norm_f32(backend, &mut output, &input, &weight, epsilon)?;

    let actual = backend.download(&output)?;
    let max_abs_error = verify(name, &actual, &expected, F32_TOLERANCE)?;

    Ok(RmsNormCaseResult {
        name,
        source: "pytorch_fixture",
        dtype: "float32",
        rows,
        hidden_size,
        epsilon,
        max_abs_error,
        tolerance: F32_TOLERANCE,
        benchmark,
        passed: true,
    })
}

fn run_rms_norm_bf16(
    stream: &Arc<Stream>,
    case: RmsNormBf16Case,
    epsilon: f32,
) -> Result<RmsNormCaseResult, BackendError> {
    let RmsNormBf16Case {
        name,
        rows,
        hidden_size,
        input_bits,
        weight_bits,
        expected_bits,
    } = case;
    check_rms_norm_shape(rows, hidden_size)?;

    let input_host = Arc::new(
        input_bits
            .into_iter()
            .map(bf16::from_bits)
            .collect::<Vec<_>>(),
    );
    let weight_host = Arc::new(
        weight_bits
            .into_iter()
            .map(bf16::from_bits)
            .collect::<Vec<_>>(),
    );
    let expected = expected_bits
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();

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
    let params = RmsNormParams::new(rows, hidden_size, epsilon);

    launch_rms_norm_bf16(stream, &mut output, &input, &weight, &params)?;
    let benchmark = benchmark_rms_norm_bf16(stream, &mut output, &input, &weight, &params)?;

    let actual = output.to_host_vec().sync_on(stream).map_err(cutile_error)?;
    let actual = actual
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let max_abs_error = verify(name, &actual, &expected, BF16_TOLERANCE)?;

    Ok(RmsNormCaseResult {
        name,
        source: "pytorch_fixture",
        dtype: "bfloat16",
        rows,
        hidden_size,
        epsilon,
        max_abs_error,
        tolerance: BF16_TOLERANCE,
        benchmark,
        passed: true,
    })
}

fn run_rope_suite(backend: &Backend, fixtures: &RopeFixtures) -> Result<RopeResult, BackendError> {
    let tables = check_tables(fixtures)?;

    // The kernel is checked with the reference tables, so its tolerance does
    // not have to absorb the difference between the two sets of tables.
    let rows = fixtures.positions.len();
    let half = fixtures.head_dim / 2;
    let cos = backend.upload(&fixtures.cos, &[rows, half])?;
    let sin = backend.upload(&fixtures.sin, &[rows, half])?;

    let mut cases = Vec::with_capacity(fixtures.cases.len());
    for case in &fixtures.cases {
        cases.push(run_rope_case(backend, fixtures, case, &cos, &sin)?);
    }

    Ok(RopeResult { tables, cases })
}

fn run_rope_case(
    backend: &Backend,
    fixtures: &RopeFixtures,
    case: &RopeCase,
    cos: &Tensor<f32>,
    sin: &Tensor<f32>,
) -> Result<RopeCaseResult, BackendError> {
    let rows = fixtures.positions.len();
    let shape = [case.heads, rows, fixtures.head_dim];
    let input = backend.upload(&case.input, &shape)?;
    let mut output = backend.zeros(&shape)?;

    backend.rope(&mut output, &input, cos, sin)?;

    let actual = backend.download(&output)?;
    let tolerance = fixtures.rotation_tolerance;
    let max_abs_error = tolerance.compare(case.name, &actual, &case.expected)?;

    Ok(RopeCaseResult {
        name: case.name,
        heads: case.heads,
        rows,
        head_dim: fixtures.head_dim,
        max_abs_error,
        tolerance,
        passed: true,
    })
}

fn run_matmul_suite(backend: &Backend) -> Result<MatmulResult, BackendError> {
    // Non-square shapes with distinct values, so a swapped or transposed
    // operand cannot produce the right answer by accident.
    Ok(MatmulResult {
        cases: vec![
            run_matmul_case(backend, "plain_2x3_times_3x4", 2, 3, 4, false)?,
            run_matmul_case(backend, "transposed_2x3_times_4x3", 2, 3, 4, true)?,
            run_kernel_matmul_kernel(backend)?,
        ],
    })
}

fn run_matmul_case(
    backend: &Backend,
    name: &'static str,
    rows: usize,
    inner: usize,
    cols: usize,
    right_transposed: bool,
) -> Result<MatmulCaseResult, BackendError> {
    let right_shape = if right_transposed {
        [cols, inner]
    } else {
        [inner, cols]
    };
    let left_host = counting(1, rows * inner);
    let right_host = counting(7, inner * cols);
    let expected = reference(&left_host, &right_host, rows, inner, cols, right_transposed);

    let left = backend.upload(&left_host, &[rows, inner])?;
    let right = backend.upload(&right_host, &right_shape)?;
    let mut output = backend.zeros(&[rows, cols])?;

    if right_transposed {
        backend.matmul_transposed(&mut output, &left, &right)?;
    } else {
        backend.matmul(&mut output, &left, &right)?;
    }

    let actual = backend.download(&output)?;
    let max_abs_error = verify(name, &actual, &expected, MATMUL_MAX_ABS_ERROR)?;

    Ok(MatmulCaseResult {
        name,
        left_shape: [rows, inner],
        right_shape,
        right_transposed,
        max_abs_error,
        tolerance: MATMUL_MAX_ABS_ERROR,
        passed: true,
    })
}

/// Runs a cuTile kernel, then cuBLAS, then a cuTile kernel again on the same
/// buffers and stream. A context, stream or pointer mismatch between the two
/// libraries shows up here as a wrong result or a CUDA error.
fn run_kernel_matmul_kernel(backend: &Backend) -> Result<MatmulCaseResult, BackendError> {
    let name = "kernel_matmul_kernel_16x16";
    let stream = &backend.stream;
    let size = KERNEL_MATMUL_SIZE;
    let elements = size * size;

    let first_host = Arc::new(cycling(17, elements));
    let second_host = Arc::new(cycling(13, elements));
    let right_host = cycling(7, elements);

    // left = first + second, computed by the vector-add kernel
    let first = api::copy_host_vec_to_device(&first_host)
        .sync_on(stream)
        .map_err(cutile_error)?;
    let second = api::copy_host_vec_to_device(&second_host)
        .sync_on(stream)
        .map_err(cutile_error)?;
    let mut left = api::zeros::<f32>(&[elements])
        .sync_on(stream)
        .map_err(cutile_error)?;
    let _ = vector_add(
        (&mut left).partition([VECTOR_TILE_ELEMENTS]),
        &first,
        &second,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;

    // product = left @ right, computed by cuBLAS on the kernel's output
    let left = left.reshape(&[size, size]).map_err(cutile_error)?;
    let right = backend.upload(&right_host, &[size, size])?;
    let mut product = backend.zeros(&[size, size])?;
    backend.matmul(&mut product, &left, &right)?;

    // doubled = product + product, computed by the kernel on cuBLAS's output
    let product = product.reshape(&[elements]).map_err(cutile_error)?;
    let mut doubled = api::zeros::<f32>(&[elements])
        .sync_on(stream)
        .map_err(cutile_error)?;
    let _ = vector_add(
        (&mut doubled).partition([VECTOR_TILE_ELEMENTS]),
        &product,
        &product,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;
    let actual = doubled
        .to_host_vec()
        .sync_on(stream)
        .map_err(cutile_error)?;

    let left_host = first_host
        .iter()
        .zip(second_host.iter())
        .map(|(first, second)| first + second)
        .collect::<Vec<_>>();
    let expected = reference(&left_host, &right_host, size, size, size, false)
        .into_iter()
        .map(|value| value * 2.0)
        .collect::<Vec<_>>();
    let max_abs_error = verify(name, &actual, &expected, MATMUL_MAX_ABS_ERROR)?;

    Ok(MatmulCaseResult {
        name,
        left_shape: [size, size],
        right_shape: [size, size],
        right_transposed: false,
        max_abs_error,
        tolerance: MATMUL_MAX_ABS_ERROR,
        passed: true,
    })
}

/// `start, start + 1, start + 2, ...` as float32.
fn counting(start: u8, elements: usize) -> Vec<f32> {
    (start..=u8::MAX).map(f32::from).take(elements).collect()
}

/// `0, 1, ..., period - 1`, repeated, as float32.
fn cycling(period: u8, elements: usize) -> Vec<f32> {
    (0..period).cycle().map(f32::from).take(elements).collect()
}

fn launch_rms_norm_bf16(
    stream: &Arc<Stream>,
    output: &mut Tensor<bf16>,
    input: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    params: &RmsNormParams,
) -> Result<(), BackendError> {
    let _ = rms_norm_bf16(
        output.partition([1, params.tile_hidden_size]),
        input,
        weight,
        params.inverse_hidden_size,
        params.epsilon,
    )
    .sync_on(stream)
    .map_err(cutile_error)?;
    Ok(())
}

fn benchmark_rms_norm_f32(
    backend: &Backend,
    output: &mut Tensor<f32>,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    epsilon: f32,
) -> Result<KernelBenchmark, BackendError> {
    for _ in 0..BENCHMARK_WARMUP_ITERATIONS {
        backend.rms_norm(output, input, weight, epsilon)?;
    }
    let timings = measure(&backend.stream, BENCHMARK_ITERATIONS, || {
        backend.rms_norm(output, input, weight, epsilon)
    })?;

    let [rows, hidden_size] = dims("rms_norm", "input", input.shape())?;
    Ok(summarize(timings, logical_bytes(rows, hidden_size, 4)))
}

fn benchmark_rms_norm_bf16(
    stream: &Arc<Stream>,
    output: &mut Tensor<bf16>,
    input: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    params: &RmsNormParams,
) -> Result<KernelBenchmark, BackendError> {
    for _ in 0..BENCHMARK_WARMUP_ITERATIONS {
        launch_rms_norm_bf16(stream, output, input, weight, params)?;
    }
    let timings = measure(stream, BENCHMARK_ITERATIONS, || {
        launch_rms_norm_bf16(stream, output, input, weight, params)
    })?;
    Ok(summarize(
        timings,
        logical_bytes(params.rows, params.hidden_size, 2),
    ))
}

fn measure(
    stream: &Arc<Stream>,
    iterations: usize,
    mut run: impl FnMut() -> Result<(), BackendError>,
) -> Result<Vec<f64>, BackendError> {
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = stream.device().new_event().map_err(cuda_error)?;
        let finished = stream.device().new_event().map_err(cuda_error)?;
        started.record(stream).map_err(cuda_error)?;
        run()?;
        finished.record(stream).map_err(cuda_error)?;
        finished.synchronize().map_err(cuda_error)?;
        timings.push(f64::from(
            started.elapsed_time(&finished).map_err(cuda_error)?,
        ));
    }
    Ok(timings)
}

fn summarize(mut timings: Vec<f64>, logical_bytes: usize) -> KernelBenchmark {
    timings.sort_by(f64::total_cmp);
    let median_ms = percentile(&timings, 50);
    let p95_ms = percentile(&timings, 95);
    let logical_bytes =
        u32::try_from(logical_bytes).expect("an RMSNorm fixture fits in four gigabytes");
    let effective_bandwidth_gbps = f64::from(logical_bytes) / (median_ms / 1_000.0) / 1.0e9;

    KernelBenchmark {
        warmup_iterations: BENCHMARK_WARMUP_ITERATIONS,
        measured_iterations: timings.len(),
        min_ms: timings[0],
        median_ms,
        p95_ms,
        effective_bandwidth_gbps,
        launches_per_iteration: 1,
    }
}

fn percentile(sorted: &[f64], percentile: usize) -> f64 {
    let index = (sorted.len() - 1) * percentile / 100;
    sorted[index]
}

fn logical_bytes(rows: usize, hidden_size: usize, element_bytes: usize) -> usize {
    // Each row reads its input and weight once, then writes one output row.
    rows * hidden_size * element_bytes * 3
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
