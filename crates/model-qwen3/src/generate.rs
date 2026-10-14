use crate::{ModelError, Result};

/// Checks a generation request before any GPU work.
///
/// The prompt and every new token must fit in the model's context window.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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
pub(crate) use gpu::generate;

#[cfg(feature = "cuda")]
mod gpu {
    use cuda_backend::Backend;

    use super::{argmax, check_request, is_finished};
    use crate::Result;
    use crate::model::Model;
    use crate::trace::Trace;

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
    ) -> Result<Vec<u32>> {
        check_request(prompt_ids.len(), max_new_tokens, model.max_positions())?;

        let (mut logits, mut caches) = model.forward(prompt_ids, &[], &mut Trace::off())?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        loop {
            let token_id = argmax(&backend.download(&logits)?);
            generated.push(token_id);
            if is_finished(&generated, max_new_tokens, eos_ids) {
                return Ok(generated);
            }
            (logits, caches) = model.forward(&[token_id], &caches, &mut Trace::off())?;
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
