use crate::generate::check_request;
use crate::{ModelError, Qwen3Artifacts, Result};

/// one run of the decode benchmark
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeBench {
    /// weights only, same as `Metrics::model_load_seconds`
    pub model_load_seconds: f64,
    pub warmup_steps: usize,
    pub contexts: Vec<ContextBench>,
    /// highest device memory seen, in bytes
    pub peak_memory_used_bytes: usize,
    pub memory_total_bytes: usize,
}

/// decode timings at one context length
#[derive(Debug, Clone, PartialEq)]
pub struct ContextBench {
    /// tokens in the cache when the timed steps start
    pub context_tokens: usize,
    /// from the second prefill, so it doesn't include compiling
    pub prefill_seconds: f64,
    /// token in to next token out, one per measured step
    pub decode_step_seconds: Vec<f64>,
    /// should be 0
    pub decode_allocations: usize,
}

/// Times decode steps at each context length, after filling the cache by
/// repeating `prompt_ids`
///
/// Use at least 16 warmup steps. cuTile recompiles the softmax for each
/// power-of-two divisor of the position (up to 16), and those compiles would
/// otherwise land in the timings
///
/// # Errors
///
/// Returns an error for an empty prompt, no measured steps, a length that
/// doesn't fit the context window, or a GPU failure
pub fn bench_decode(
    artifacts: &Qwen3Artifacts,
    prompt_ids: &[u32],
    context_lengths: &[usize],
    warmup_steps: usize,
    measured_steps: usize,
) -> Result<DecodeBench> {
    if prompt_ids.is_empty() {
        return Err(ModelError::EmptyPrompt);
    }
    check_bench(
        context_lengths,
        warmup_steps,
        measured_steps,
        artifacts.config.max_position_embeddings,
    )?;

    bench_on_gpu(
        artifacts,
        prompt_ids,
        context_lengths,
        warmup_steps,
        measured_steps,
    )
}

/// Checks the benchmark settings before any GPU work.
///
/// The warmup and the timed steps run on separate caches, so each cache needs
/// room for whichever is longer, not both.
fn check_bench(
    context_lengths: &[usize],
    warmup_steps: usize,
    measured_steps: usize,
    max_positions: usize,
) -> Result<()> {
    if measured_steps == 0 {
        return Err(ModelError::NoMeasuredSteps);
    }
    for &context_tokens in context_lengths {
        check_request(
            context_tokens,
            warmup_steps.max(measured_steps),
            max_positions,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_on_gpu(
    artifacts: &Qwen3Artifacts,
    prompt_ids: &[u32],
    context_lengths: &[usize],
    warmup_steps: usize,
    measured_steps: usize,
) -> Result<DecodeBench> {
    use std::time::Instant;

    use crate::generate::argmax;
    use crate::model::Model;
    use crate::trace::Trace;

    let backend = cuda_backend::Backend::new()?;
    let load_started = Instant::now();
    let model = Model::load(&backend, artifacts)?;
    let model_load_seconds = load_started.elapsed().as_secs_f64();

    let mut contexts = Vec::with_capacity(context_lengths.len());
    let mut peak_memory_used_bytes = 0;
    let mut memory_total_bytes = 0;
    for &context_tokens in context_lengths {
        let prompt = prompt_ids
            .iter()
            .copied()
            .cycle()
            .take(context_tokens)
            .collect::<Vec<_>>();
        // check_bench made sure this fits in the context window
        let capacity = context_tokens + warmup_steps.max(measured_steps);
        let mut prefill_workspace = model.new_workspace(context_tokens, capacity)?;
        let mut workspace = model.new_workspace(1, capacity)?;

        // compile every kernel for this shape on a cache that gets thrown
        // away, including the softmax versions the first decode steps need
        {
            let mut caches = model.new_caches(capacity)?;
            let logits = model.prefill(
                &prompt,
                &mut caches,
                &mut prefill_workspace,
                &mut Trace::off(),
            )?;
            let mut token_id = argmax(logits);
            for _ in 0..warmup_steps {
                let logits =
                    model.decode_step(token_id, &mut caches, &mut workspace, &mut Trace::off())?;
                token_id = argmax(logits);
            }
        }

        // a fresh cache, so the timed steps start from exactly `context_tokens`
        let mut caches = model.new_caches(capacity)?;
        let prefill_started = Instant::now();
        let logits = model.prefill(
            &prompt,
            &mut caches,
            &mut prefill_workspace,
            &mut Trace::off(),
        )?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        let mut token_id = argmax(logits);

        let (memory_used, memory_total) = backend.memory_in_use()?;
        peak_memory_used_bytes = peak_memory_used_bytes.max(memory_used);
        memory_total_bytes = memory_total;
        drop(prefill_workspace);

        let mut decode_step_seconds = Vec::with_capacity(measured_steps);
        let allocations_before = backend.allocations();
        for _ in 0..measured_steps {
            let step_started = Instant::now();
            let logits =
                model.decode_step(token_id, &mut caches, &mut workspace, &mut Trace::off())?;
            token_id = argmax(logits);
            decode_step_seconds.push(step_started.elapsed().as_secs_f64());
        }

        contexts.push(ContextBench {
            context_tokens,
            prefill_seconds,
            decode_step_seconds,
            decode_allocations: backend.allocations() - allocations_before,
        });
    }

    Ok(DecodeBench {
        model_load_seconds,
        warmup_steps,
        contexts,
        peak_memory_used_bytes,
        memory_total_bytes,
    })
}

#[cfg(not(feature = "cuda"))]
fn bench_on_gpu(
    _artifacts: &Qwen3Artifacts,
    _prompt_ids: &[u32],
    _context_lengths: &[usize],
    _warmup_steps: usize,
    _measured_steps: usize,
) -> Result<DecodeBench> {
    Err(cuda_backend::BackendError::CudaDisabled.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_default_settings() {
        assert!(check_bench(&[128, 512, 2048], 16, 32, 32_768).is_ok());
    }

    #[test]
    fn each_cache_only_needs_room_for_the_longer_run() {
        // 32_700 + 64 fits, but 32_700 + 16 + 64 would not
        assert!(check_bench(&[32_700], 16, 64, 32_768).is_ok());
        assert!(check_bench(&[32_705], 16, 64, 32_768).is_err());
    }

    #[test]
    fn rejects_settings_that_would_overflow() {
        assert!(check_bench(&[128], usize::MAX, 32, 32_768).is_err());
        assert!(check_bench(&[128], 16, usize::MAX, 32_768).is_err());
        assert!(check_bench(&[usize::MAX], 16, 32, 32_768).is_err());
    }

    #[test]
    fn rejects_no_measured_steps() {
        assert!(matches!(
            check_bench(&[128], 16, 0, 32_768),
            Err(ModelError::NoMeasuredSteps)
        ));
    }
}
