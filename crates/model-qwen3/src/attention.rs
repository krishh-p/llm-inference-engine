use cuda_backend::{Backend, Tensor};

use crate::kv_cache::KvCache;
use crate::trace::Trace;
use crate::weights::AttentionWeights;
use crate::workspace::AttentionBuffers;
use crate::{Qwen3Config, Result};

/// One layer's attention with its weights on the GPU.
pub(crate) struct Attention<'a> {
    backend: &'a Backend,
    head_dim: usize,
    epsilon: f32,
    q_proj: Tensor<f32>,
    k_proj: Tensor<f32>,
    v_proj: Tensor<f32>,
    o_proj: Tensor<f32>,
    q_norm: Tensor<f32>,
    k_norm: Tensor<f32>,
}

impl<'a> Attention<'a> {
    pub(crate) fn new(
        backend: &'a Backend,
        config: &Qwen3Config,
        weights: &AttentionWeights,
    ) -> Result<Self> {
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        Ok(Self {
            backend,
            head_dim,
            epsilon: config.rms_norm_eps,
            q_proj: backend.upload(&weights.q_proj, &[heads * head_dim, hidden_size])?,
            k_proj: backend.upload(&weights.k_proj, &[kv_heads * head_dim, hidden_size])?,
            v_proj: backend.upload(&weights.v_proj, &[kv_heads * head_dim, hidden_size])?,
            o_proj: backend.upload(&weights.o_proj, &[hidden_size, heads * head_dim])?,
            q_norm: backend.upload(&weights.q_norm, &[head_dim])?,
            k_norm: backend.upload(&weights.k_norm, &[head_dim])?,
        })
    }

    /// Runs attention for `rows` new tokens, which come straight after the
    /// tokens in `cache`. Each new token sees the cached tokens, the new
    /// tokens before it, and itself.
    ///
    /// `x` is the normalized layer input, `[rows, hidden_size]`, and the
    /// result is written to `output`, which has the same shape. The new
    /// tokens' keys and values are written into `cache`, which is empty for
    /// prefill. Everything in between is written to `buffers`.
    pub(crate) fn forward(
        &self,
        x: &Tensor<f32>,
        output: &mut Tensor<f32>,
        rows: usize,
        cache: &mut KvCache,
        buffers: &mut AttentionBuffers,
        trace: &mut Trace,
    ) -> Result<()> {
        let backend = self.backend;
        let b = buffers;
        let cached = cache.len;
        let total = cached + rows;

        // project, then split into heads: [rows, heads * head_dim] to [heads, rows, head_dim]
        backend.matmul_transposed(&mut b.q_projected, x, &self.q_proj)?;
        backend.matmul_transposed(&mut b.k_projected, x, &self.k_proj)?;
        backend.matmul_transposed(&mut b.v_projected, x, &self.v_proj)?;
        backend.split_heads(&mut b.q, &b.q_projected)?;
        backend.split_heads(&mut b.k, &b.k_projected)?;
        backend.split_heads(&mut b.v, &b.v_projected)?;
        trace.save(backend, "q", &b.q)?;
        trace.save(backend, "k", &b.k)?;
        trace.save(backend, "v", &b.v)?;

        // positions continue from the end of the cache
        backend.copy_rows(&mut b.cos, 0, &b.cos_table, cached, rows)?;
        backend.copy_rows(&mut b.sin, 0, &b.sin_table, cached, rows)?;
        backend.rms_norm_heads(&mut b.q_normalized, &b.q, &self.q_norm, self.epsilon)?;
        backend.rms_norm_heads(&mut b.k_normalized, &b.k, &self.k_norm, self.epsilon)?;
        backend.rope(&mut b.q_rotated, &b.q_normalized, &b.cos, &b.sin)?;
        backend.rope(&mut b.k_rotated, &b.k_normalized, &b.cos, &b.sin)?;
        trace.save(backend, "q_rope", &b.q_rotated)?;
        trace.save(backend, "k_rope", &b.k_rotated)?;

        // the new keys and values go after the cached ones
        if cached > 0 {
            trace.save_rows(backend, "cache_k_in", &cache.k, cached)?;
            trace.save_rows(backend, "cache_v_in", &cache.v, cached)?;
        }
        cache.append(backend, &b.k_rotated, &b.v, rows)?;
        trace.save_rows(backend, "cache_k_out", &cache.k, total)?;
        trace.save_rows(backend, "cache_v_out", &cache.v, total)?;

        // rows are the new tokens, columns are the tokens they attend to
        backend.attention_scores(
            &mut b.scores,
            &b.q_rotated,
            &cache.k,
            total,
            self.score_scale(),
        )?;
        trace.save_columns(backend, "scores", &b.scores, total)?;
        backend.causal_softmax(&mut b.probs, &b.scores, cached)?;
        trace.save_columns(backend, "attention_probs", &b.probs, total)?;

        // weighted sum of values, then the heads go back together
        backend.attention_mix(&mut b.mixed, &b.probs, &cache.v, total)?;
        backend.merge_heads(&mut b.merged, &b.mixed)?;
        backend.matmul_transposed(output, &b.merged, &self.o_proj)?;
        trace.save(backend, "attention_output", output)?;
        Ok(())
    }

    /// `1 / sqrt(head_dim)`, which keeps the scores from growing with the head size.
    fn score_scale(&self) -> f32 {
        let head_dim = u16::try_from(self.head_dim).expect("head_dim fits in u16");
        f32::from(head_dim).sqrt().recip()
    }
}
