use crate::{ModelError, Qwen3Artifacts, Result};

/// One prompt's greedy completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub prompt_ids: Vec<u32>,
    /// The generated tokens, ending with the end-of-sequence token if one
    /// stopped generation.
    pub token_ids: Vec<u32>,
    /// The generated tokens as text, without special tokens.
    pub text: String,
}

/// Loads the model on the GPU and greedily completes `prompt`.
///
/// Generation stops at the end-of-sequence token or after `max_new_tokens`.
/// The request is checked before the model is loaded, so a prompt that is
/// empty or too long fails straight away.
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
) -> Result<Completion> {
    let config = &artifacts.config;
    let prompt_ids = artifacts.tokenizer.encode(prompt)?;
    check_request(
        prompt_ids.len(),
        max_new_tokens,
        config.max_position_embeddings,
    )?;

    let token_ids = complete_on_gpu(artifacts, &prompt_ids, max_new_tokens)?;
    let text = artifacts.tokenizer.decode(&token_ids)?;
    Ok(Completion {
        prompt_ids,
        token_ids,
        text,
    })
}

#[cfg(feature = "cuda")]
fn complete_on_gpu(
    artifacts: &Qwen3Artifacts,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<Vec<u32>> {
    use crate::model::Model;

    let backend = cuda_backend::Backend::new()?;
    let model = Model::load(&backend, artifacts)?;
    let eos_ids = [artifacts.config.eos_token_id];
    let generated = generate(&backend, &model, prompt_ids, max_new_tokens, &eos_ids)?;
    Ok(generated.token_ids)
}

#[cfg(not(feature = "cuda"))]
fn complete_on_gpu(
    _artifacts: &Qwen3Artifacts,
    _prompt_ids: &[u32],
    _max_new_tokens: usize,
) -> Result<Vec<u32>> {
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
    use cuda_backend::Backend;

    use super::{argmax, check_request, is_finished};
    use crate::Result;
    use crate::model::Model;
    use crate::trace::Trace;

    /// The tokens one greedy generation produced.
    pub(crate) struct Generated {
        pub token_ids: Vec<u32>,
        /// Buffers allocated while decoding. Should be 0, since every step
        /// reuses the same workspace
        pub decode_allocations: usize,
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

        // the caches are allocated once, with room for the whole sequence
        let capacity = prompt_ids.len() + max_new_tokens;
        let mut caches = model.new_caches(capacity)?;

        // prefill's workspace is sized for the whole prompt and used once
        let mut prefill_workspace = model.new_workspace(prompt_ids.len(), capacity)?;
        let logits = model.prefill(
            prompt_ids,
            &mut caches,
            &mut prefill_workspace,
            &mut Trace::off(),
        )?;
        let mut token_id = argmax(logits);
        drop(prefill_workspace);

        // decode's is sized for one token and reused for every step
        let mut workspace = model.new_workspace(1, capacity)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let allocations_before = backend.allocations();
        loop {
            generated.push(token_id);
            if is_finished(&generated, max_new_tokens, eos_ids) {
                return Ok(Generated {
                    token_ids: generated,
                    decode_allocations: backend.allocations() - allocations_before,
                });
            }
            let logits =
                model.decode_step(token_id, &mut caches, &mut workspace, &mut Trace::off())?;
            token_id = argmax(logits);
        }
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
