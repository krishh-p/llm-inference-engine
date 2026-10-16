use model_qwen3::{Completion, DecodeBench};
use serde_json::{Value, json};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One generation run as a JSON object.
///
/// Fields that both engines measure use the same names as the Python
/// baseline in `modal_apps/dev.py`, so runs can be compared field by field.
pub(crate) fn run_json(prompt: &str, completion: &Completion, seed: u64, dtype: &str) -> Value {
    let metrics = &completion.metrics;
    let prompt_tokens = completion.prompt_ids.len();
    let generated_tokens = completion.token_ids.len();
    let decode_steps = metrics.decode_step_seconds.len();
    let decode_seconds = metrics.decode_step_seconds.iter().sum();

    json!({
        "engine": "rust",
        "dtype": dtype,
        "seed": seed,
        "prompt": prompt,
        "completion": completion.text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "token_ids": completion.token_ids,
        "warmup_runs": metrics.warmup_runs,
        "model_load_seconds": metrics.model_load_seconds,
        "cold_time_to_first_token_seconds": metrics.cold_time_to_first_token_seconds,
        "prefill_seconds": metrics.prefill_seconds,
        "prompt_tokens_per_second": per_second(prompt_tokens, metrics.prefill_seconds),
        "time_to_first_token_seconds": metrics.time_to_first_token_seconds,
        "decode_step_median_ms": percentile(&metrics.decode_step_seconds, 50).map(milliseconds),
        "decode_step_p95_ms": percentile(&metrics.decode_step_seconds, 95).map(milliseconds),
        "decode_tokens_per_second": per_second(decode_steps, decode_seconds),
        "host_overhead_per_token_ms": percentile(&metrics.host_seconds, 50).map(milliseconds),
        "generation_seconds": metrics.generation_seconds,
        "tokens_per_second": per_second(generated_tokens, metrics.generation_seconds),
        "gpu_memory_used_gib": gib(metrics.peak_memory_used_bytes),
        "gpu_memory_total_gib": gib(metrics.memory_total_bytes),
    })
}

/// One decode benchmark run as a JSON object, with a summary per context length.
pub(crate) fn bench_json(bench: &DecodeBench, dtype: &str) -> Value {
    let contexts = bench
        .contexts
        .iter()
        .map(|context| {
            let steps = &context.decode_step_seconds;
            json!({
                "context_tokens": context.context_tokens,
                "prefill_seconds": context.prefill_seconds,
                "prompt_tokens_per_second": per_second(context.context_tokens, context.prefill_seconds),
                "measured_steps": steps.len(),
                "decode_step_min_ms": percentile(steps, 0).map(milliseconds),
                "decode_step_median_ms": percentile(steps, 50).map(milliseconds),
                "decode_step_p95_ms": percentile(steps, 95).map(milliseconds),
                "decode_step_max_ms": percentile(steps, 100).map(milliseconds),
                "decode_tokens_per_second": per_second(steps.len(), steps.iter().sum()),
                "decode_allocations": context.decode_allocations,
                "kernel_launches_per_step": per_step(context.decode_kernel_launches, steps.len()),
                "cublas_calls_per_step": per_step(context.decode_cublas_calls, steps.len()),
                "host_waits_per_step": per_step(context.decode_host_waits, steps.len()),
                "decode_step_ms": steps.iter().copied().map(milliseconds).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "engine": "rust",
        "dtype": dtype,
        "warmup_steps": bench.warmup_steps,
        "model_load_seconds": bench.model_load_seconds,
        "gpu_memory_used_gib": gib(bench.peak_memory_used_bytes),
        "gpu_memory_total_gib": gib(bench.memory_total_bytes),
        "contexts": contexts,
    })
}

/// The value that `percent` of the values are at or below, by nearest rank.
/// `None` for an empty list.
fn percentile(values: &[f64], percent: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    // the smallest rank that covers `percent` of the values
    let rank = (percent * sorted.len()).div_ceil(100).max(1);
    Some(sorted[rank - 1])
}

// token counts and byte sizes are far below 2^52, so they convert exactly
#[allow(clippy::cast_precision_loss)]
fn per_second(count: usize, seconds: f64) -> Option<f64> {
    (count > 0 && seconds > 0.0).then(|| count as f64 / seconds)
}

// every step runs the same operations, so a fraction would mean the steps differed
#[allow(clippy::cast_precision_loss)]
fn per_step(count: usize, steps: usize) -> Option<f64> {
    (steps > 0).then(|| count as f64 / steps as f64)
}

#[allow(clippy::cast_precision_loss)]
fn gib(bytes: usize) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

fn milliseconds(seconds: f64) -> f64 {
    seconds * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_the_nearest_rank() {
        let values = (1..=20).map(f64::from).collect::<Vec<_>>();

        assert_eq!(percentile(&values, 50), Some(10.0));
        assert_eq!(percentile(&values, 95), Some(19.0));
        assert_eq!(percentile(&values, 100), Some(20.0));
    }

    #[test]
    fn percentile_sorts_first() {
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 50), Some(2.0));
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 95), Some(3.0));
    }

    #[test]
    fn percentile_of_nothing_is_none() {
        assert_eq!(percentile(&[], 50), None);
        assert_eq!(percentile(&[4.0], 0), Some(4.0));
    }

    #[test]
    fn rates_need_tokens_and_time() {
        assert_eq!(per_second(10, 2.0), Some(5.0));
        assert_eq!(per_second(0, 2.0), None);
        assert_eq!(per_second(10, 0.0), None);
    }

    #[test]
    fn counts_are_averaged_over_the_steps() {
        assert_eq!(per_step(730, 2), Some(365.0));
        assert_eq!(per_step(731, 2), Some(365.5));
        assert_eq!(per_step(5, 0), None);
    }
}
