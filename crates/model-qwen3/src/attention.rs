use cuda_backend::{Backend, Tensor, reshape, rope_tables};

use crate::kv_cache::KvCache;
use crate::trace::Trace;
use crate::weights::AttentionWeights;
use crate::{Qwen3Config, Result};

/// One layer's attention with its weights on the GPU.
pub(crate) struct Attention<'a> {
    backend: &'a Backend,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    epsilon: f32,
    theta: f32,
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
            heads,
            kv_heads,
            head_dim,
            hidden_size,
            epsilon: config.rms_norm_eps,
            theta: config.rope_theta,
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
    /// result has the same shape. The new tokens' keys and values are written
    /// into `cache`, which is empty for prefill.
    pub(crate) fn forward(
        &self,
        x: &Tensor<f32>,
        rows: usize,
        cache: &mut KvCache,
        trace: &mut Trace,
    ) -> Result<Tensor<f32>> {
        let backend = self.backend;
        let cached = cache.len;
        let total = cached + rows;
        let (q, k, v) = self.project(x, rows, trace)?;

        // positions continue from the end of the cache
        let positions = (cached..total)
            .map(|position| u32::try_from(position).expect("positions fit in u32"))
            .collect::<Vec<_>>();
        let half = self.head_dim / 2;
        let (cos, sin) = rope_tables(&positions, self.head_dim, self.theta);
        let cos = backend.upload(&cos, &[rows, half])?;
        let sin = backend.upload(&sin, &[rows, half])?;
        let q = self.normalize_and_rotate(q, &self.q_norm, self.heads, rows, &cos, &sin)?;
        let k = self.normalize_and_rotate(k, &self.k_norm, self.kv_heads, rows, &cos, &sin)?;
        trace.save(backend, "q_rope", &q)?;
        trace.save(backend, "k_rope", &k)?;

        // the new keys and values go after the cached ones
        if cached > 0 {
            trace.save_rows(backend, "cache_k_in", &cache.k, cached)?;
            trace.save_rows(backend, "cache_v_in", &cache.v, cached)?;
        }
        cache.append(backend, &k, &v, rows)?;
        trace.save_rows(backend, "cache_k_out", &cache.k, total)?;
        trace.save_rows(backend, "cache_v_out", &cache.v, total)?;

        // rows are the new tokens, columns are the tokens they attend to
        let mut scores = backend.zeros(&[self.heads, rows, total])?;
        backend.matmul_heads_transposed(&mut scores, &q, &cache.k, total, self.score_scale())?;
        trace.save(backend, "scores", &scores)?;

        let mut probs = backend.zeros(&[self.heads, rows, total])?;
        backend.causal_softmax(&mut probs, &scores, cached)?;
        trace.save(backend, "attention_probs", &probs)?;

        // weighted sum of values, [heads, rows, head_dim]
        let mut mixed = backend.zeros(&[self.heads, rows, self.head_dim])?;
        backend.matmul_heads(&mut mixed, &probs, &cache.v, total)?;

        let merged = self.merge_heads(&mixed, rows)?;
        let mut output = backend.zeros(&[rows, self.hidden_size])?;
        backend.matmul_transposed(&mut output, &merged, &self.o_proj)?;
        trace.save(backend, "attention_output", &output)?;
        Ok(output)
    }

    /// Projects the layer input into query, key and value heads.
    fn project(
        &self,
        x: &Tensor<f32>,
        rows: usize,
        trace: &mut Trace,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        let q = self.project_heads(x, &self.q_proj, rows, self.heads)?;
        let k = self.project_heads(x, &self.k_proj, rows, self.kv_heads)?;
        let v = self.project_heads(x, &self.v_proj, rows, self.kv_heads)?;
        trace.save(self.backend, "q", &q)?;
        trace.save(self.backend, "k", &k)?;
        trace.save(self.backend, "v", &v)?;
        Ok((q, k, v))
    }

    /// `x @ weight^T`, split into heads: `[rows, hidden_size]` to `[heads, rows, head_dim]`.
    fn project_heads(
        &self,
        x: &Tensor<f32>,
        weight: &Tensor<f32>,
        rows: usize,
        heads: usize,
    ) -> Result<Tensor<f32>> {
        let backend = self.backend;
        let mut projected = backend.zeros(&[rows, heads * self.head_dim])?;
        backend.matmul_transposed(&mut projected, x, weight)?;

        let by_token = reshape(projected, &[rows, heads, self.head_dim])?;
        let mut by_head = backend.zeros(&[heads, rows, self.head_dim])?;
        backend.swap_leading_axes(&mut by_head, &by_token)?;
        Ok(by_head)
    }

    /// `[heads, rows, head_dim]` back to `[rows, heads * head_dim]`.
    fn merge_heads(&self, x: &Tensor<f32>, rows: usize) -> Result<Tensor<f32>> {
        let mut by_token = self.backend.zeros(&[rows, self.heads, self.head_dim])?;
        self.backend.swap_leading_axes(&mut by_token, x)?;
        Ok(reshape(by_token, &[rows, self.heads * self.head_dim])?)
    }

    /// Applies the per-head `RMSNorm` and then `RoPE` to `[heads, rows, head_dim]`.
    fn normalize_and_rotate(
        &self,
        x: Tensor<f32>,
        norm_weight: &Tensor<f32>,
        heads: usize,
        rows: usize,
        cos: &Tensor<f32>,
        sin: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        let backend = self.backend;
        let shape = [heads, rows, self.head_dim];
        let vectors = heads * rows;

        // the norm treats every head of every token as its own row
        let x = reshape(x, &[vectors, self.head_dim])?;
        let mut normalized = backend.zeros(&[vectors, self.head_dim])?;
        backend.rms_norm(&mut normalized, &x, norm_weight, self.epsilon)?;

        let normalized = reshape(normalized, &shape)?;
        let mut rotated = backend.zeros(&shape)?;
        backend.rope(&mut rotated, &normalized, cos, sin)?;
        Ok(rotated)
    }

    /// `1 / sqrt(head_dim)`, which keeps the scores from growing with the head size.
    fn score_scale(&self) -> f32 {
        let head_dim = u16::try_from(self.head_dim).expect("head_dim fits in u16");
        f32::from(head_dim).sqrt().recip()
    }
}
