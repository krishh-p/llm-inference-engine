use crate::{ModelError, Qwen3Artifacts, Result};

/// One prompt's greedy completion.
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub prompt_ids: Vec<u32>,
    /// The generated tokens, ending with the end-of-sequence token if one
    /// stopped generation.
    pub token_ids: Vec<u32>,
    /// The generated tokens as text, without special tokens.
    pub text: String,
    pub metrics: Metrics,
}

/// How long each part of a completion took, and how much device memory it used.
///
/// Times are host wall-clock seconds. Every GPU call waits for its work to
/// finish, so they include the GPU time too.
#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    /// Reading every weight, widening it to float32 and uploading it. Opening
    /// the tokenizer and setting up CUDA are not counted.
    pub model_load_seconds: f64,
    /// Generations run before the measured one.
    pub warmup_runs: usize,
    /// Time to first token of the first warmup run, which also compiles
    /// every GPU kernel. `None` without warmup runs.
    pub cold_time_to_first_token_seconds: Option<f64>,
    /// Running the prompt through the model, up to its logits on the host.
    pub prefill_seconds: f64,
    /// From the start of generation to the first new token, including
    /// allocating the caches.
    pub time_to_first_token_seconds: f64,
    /// Each decode step, from passing in a token to having the next one.
    pub decode_step_seconds: Vec<f64>,
    /// The part of each decode step spent on the host after the logits
    /// arrive, picking the next token.
    pub host_seconds: Vec<f64>,
    /// From the start of generation to the last new token.
    pub generation_seconds: f64,
    /// The most device memory in use at the points it was sampled, in bytes.
    pub peak_memory_used_bytes: usize,
    pub memory_total_bytes: usize,
}

/// Loads the model on the GPU and greedily completes `prompt`.
///
/// Generation stops at the end-of-sequence token or after `max_new_tokens`.
/// `warmup_runs` complete the prompt first without being measured, since the
/// first run also compiles the GPU kernels. The request is checked before the
/// model is loaded, so a prompt that is empty or too long fails straight away.
///
/// # Errors
///
/// Returns an error when the prompt cannot be tokenized, the request does not
/// fit in the context window, a weight cannot be loaded, or the GPU is
/// unavailable.
pub fn complete(
    artifacts: &Qwen3Artifacts,
    prompt: &str,
    max_new_tokens: usize,
    warmup_runs: usize,
) -> Result<Completion> {
    let config = &artifacts.config;
    let prompt_ids = artifacts.tokenizer.encode(prompt)?;
    check_request(
        prompt_ids.len(),
        max_new_tokens,
        config.max_position_embeddings,
    )?;

    let (token_ids, metrics) =
        complete_on_gpu(artifacts, &prompt_ids, max_new_tokens, warmup_runs)?;
    let text = artifacts.tokenizer.decode(&token_ids)?;
    Ok(Completion {
        prompt_ids,
        token_ids,
        text,
        metrics,
    })
}

#[cfg(feature = "cuda")]
fn complete_on_gpu(
    artifacts: &Qwen3Artifacts,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    warmup_runs: usize,
) -> Result<(Vec<u32>, Metrics)> {
    use std::time::Instant;

    use crate::model::Model;

    // load time covers the weights only, the same as the Python baseline
    let backend = cuda_backend::Backend::new()?;
    let load_started = Instant::now();
    let model = Model::load(&backend, artifacts)?;
    let model_load_seconds = load_started.elapsed().as_secs_f64();

    let eos_ids = [artifacts.config.eos_token_id];
    let mut cold_time_to_first_token_seconds = None;
    for _ in 0..warmup_runs {
        let warmup = generate(&backend, &model, prompt_ids, max_new_tokens, &eos_ids)?;
        cold_time_to_first_token_seconds.get_or_insert(warmup.time_to_first_token_seconds);
    }

    let generated = generate(&backend, &model, prompt_ids, max_new_tokens, &eos_ids)?;
    let metrics = Metrics {
        model_load_seconds,
        warmup_runs,
        cold_time_to_first_token_seconds,
        prefill_seconds: generated.prefill_seconds,
        time_to_first_token_seconds: generated.time_to_first_token_seconds,
        decode_step_seconds: generated.decode_step_seconds,
        host_seconds: generated.host_seconds,
        generation_seconds: generated.generation_seconds,
        peak_memory_used_bytes: generated.peak_memory_used_bytes,
        memory_total_bytes: generated.memory_total_bytes,
    };
    Ok((generated.token_ids, metrics))
}

#[cfg(not(feature = "cuda"))]
fn complete_on_gpu(
    _artifacts: &Qwen3Artifacts,
    _prompt_ids: &[u32],
    _max_new_tokens: usize,
    _warmup_runs: usize,
) -> Result<(Vec<u32>, Metrics)> {
    Err(cuda_backend::BackendError::CudaDisabled.into())
}

/// Checks a generation request before any GPU work.
///
/// The prompt and every new token must fit in the model's context window.
pub(crate) fn check_request(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_positions: usize,
) -> Result<()> {
    if prompt_tokens == 0 {
        return Err(ModelError::EmptyPrompt);
    }
    if max_new_tokens == 0 {
        return Err(ModelError::NoNewTokens);
    }
    if max_new_tokens > max_positions || prompt_tokens > max_positions - max_new_tokens {
        return Err(ModelError::ContextTooLong {
            prompt_tokens,
            max_new_tokens,
            max_positions,
        });
    }
    Ok(())
}

/// The index of the largest logit, which is the greedy choice of next token.
/// Ties go to the lowest index, the same as `torch.argmax`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).expect("vocabulary indices fit in u32")
}

/// Generation stops after an end-of-sequence token or once enough tokens are out.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn is_finished(generated: &[u32], max_new_tokens: usize, eos_ids: &[u32]) -> bool {
    generated.len() >= max_new_tokens
        || generated
            .last()
            .is_some_and(|token_id| eos_ids.contains(token_id))
}

#[cfg(feature = "cuda")]
pub(crate) use gpu::{generate, generate_uncached};

#[cfg(feature = "cuda")]
mod gpu {
    use std::time::Instant;

    use cuda_backend::Backend;

    use super::{argmax, check_request, is_finished};
    use crate::Result;
    use crate::model::Model;
    use crate::trace::Trace;

    /// The tokens one greedy generation produced, and how long it took.
    /// See [`Metrics`](super::Metrics) for what each time covers.
    pub(crate) struct Generated {
        pub token_ids: Vec<u32>,
        /// Buffers allocated while decoding. Should be 0, since every step
        /// reuses the same workspace
        pub decode_allocations: usize,
        pub prefill_seconds: f64,
        pub time_to_first_token_seconds: f64,
        pub decode_step_seconds: Vec<f64>,
        pub host_seconds: Vec<f64>,
        pub generation_seconds: f64,
        pub peak_memory_used_bytes: usize,
        pub memory_total_bytes: usize,
    }

    /// Greedily generates up to `max_new_tokens` tokens after the prompt.
    ///
    /// The prompt runs once as prefill. After that, each new token runs on
    /// its own against the caches of everything before it. The result
    /// includes the end-of-sequence token when one stops generation.
    pub(crate) fn generate(
        backend: &Backend,
        model: &Model,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_ids: &[u32],
    ) -> Result<Generated> {
        check_request(prompt_ids.len(), max_new_tokens, model.max_positions())?;
        let started = Instant::now();

        // the caches are allocated once, with room for the whole sequence
        let capacity = prompt_ids.len() + max_new_tokens;
        let mut caches = model.new_caches(capacity)?;

        // prefill's workspace is sized for the whole prompt and used once
        let mut prefill_workspace = model.new_workspace(prompt_ids.len(), capacity)?;
        let prefill_started = Instant::now();
        let logits = model.prefill(
            prompt_ids,
            &mut caches,
            &mut prefill_workspace,
            &mut Trace::off(),
        )?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        let mut token_id = argmax(logits);
        let time_to_first_token_seconds = started.elapsed().as_secs_f64();
        // sampled while the prefill workspace still exists, when memory is highest
        let (memory_after_prefill, memory_total_bytes) = backend.memory_in_use()?;
        drop(prefill_workspace);

        // decode's is sized for one token and reused for every step
        let mut workspace = model.new_workspace(1, capacity)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        // sized up front too, so timing the steps doesn't allocate
        let mut decode_step_seconds = Vec::with_capacity(max_new_tokens);
        let mut host_seconds = Vec::with_capacity(max_new_tokens);
        let allocations_before = backend.allocations();
        loop {
            generated.push(token_id);
            if is_finished(&generated, max_new_tokens, eos_ids) {
                break;
            }
            let step_started = Instant::now();
            let logits =
                model.decode_step(token_id, &mut caches, &mut workspace, &mut Trace::off())?;
            let logits_arrived = Instant::now();
            token_id = argmax(logits);
            decode_step_seconds.push(step_started.elapsed().as_secs_f64());
            host_seconds.push(logits_arrived.elapsed().as_secs_f64());
        }
        let generation_seconds = started.elapsed().as_secs_f64();
        let decode_allocations = backend.allocations() - allocations_before;
        let (memory_at_end, _) = backend.memory_in_use()?;

        Ok(Generated {
            token_ids: generated,
            decode_allocations,
            prefill_seconds,
            time_to_first_token_seconds,
            decode_step_seconds,
            host_seconds,
            generation_seconds,
            peak_memory_used_bytes: memory_after_prefill.max(memory_at_end),
            memory_total_bytes,
        })
    }

    /// Greedily generates like [`generate`], but runs the whole sequence again
    /// for every new token instead of keeping caches.
    ///
    /// Much slower, and only here to check the cached path against.
    pub(crate) fn generate_uncached(
        model: &Model,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_ids: &[u32],
    ) -> Result<Vec<u32>> {
        check_request(prompt_ids.len(), max_new_tokens, model.max_positions())?;

        let mut sequence = prompt_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        loop {
            // fresh caches every time, so nothing carries over between steps
            let rows = sequence.len();
            let mut caches = model.new_caches(rows)?;
            let mut workspace = model.new_workspace(rows, rows)?;
            let logits =
                model.prefill(&sequence, &mut caches, &mut workspace, &mut Trace::off())?;
            let token_id = argmax(logits);
            generated.push(token_id);
            if is_finished(&generated, max_new_tokens, eos_ids) {
                return Ok(generated);
            }
            sequence.push(token_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_largest_logit() {
        assert_eq!(argmax(&[0.5, -1.0, 3.0, 2.0]), 2);
    }

    #[test]
    fn argmax_breaks_ties_toward_the_first_index() {
        assert_eq!(argmax(&[1.0, 4.0, 4.0]), 1);
    }

    #[test]
    fn stops_at_the_token_limit() {
        assert!(!is_finished(&[5, 6], 3, &[0]));
        assert!(is_finished(&[5, 6, 7], 3, &[0]));
    }

    #[test]
    fn stops_at_an_end_of_sequence_token() {
        assert!(is_finished(&[5, 0], 8, &[0]));
        assert!(!is_finished(&[0, 5], 8, &[0]));
    }

    #[test]
    fn accepts_a_request_that_fills_the_context() {
        assert!(check_request(30, 2, 32).is_ok());
    }

    #[test]
    fn rejects_a_request_past_the_context() {
        let error = check_request(31, 2, 32).unwrap_err();

        assert_eq!(
            error.to_string(),
            "31 prompt tokens plus 2 new tokens do not fit in 32 positions"
        );
        assert!(check_request(1, usize::MAX, 32).is_err());
    }

    #[test]
    fn rejects_an_empty_prompt_or_no_new_tokens() {
        assert!(matches!(
            check_request(0, 8, 32),
            Err(ModelError::EmptyPrompt)
        ));
        assert!(matches!(
            check_request(5, 0, 32),
            Err(ModelError::NoNewTokens)
        ));
    }
}
