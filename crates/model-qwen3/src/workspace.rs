use cuda_backend::{Backend, Tensor, rope_tables};

use crate::{ModelError, Qwen3Config, Result};

/// Every tensor one forward pass writes, allocated once for a fixed number
/// of new tokens and a fixed cache capacity.
///
/// Prefill and decode each get their own workspace. Prefill's is sized for
/// the whole prompt and used once. Decode's is sized for one token and reused
/// for every step, so the token loop allocates nothing. The layers run one
/// after another, so they all share the same buffers.
pub(crate) struct Workspace {
    /// How many new tokens each forward pass runs.
    pub rows: usize,
    /// The hidden state of the new tokens, `[rows, hidden_size]`. It goes in
    /// as the embeddings and is updated in place by every layer.
    pub x: Tensor<f32>,
    pub block: BlockBuffers,
    pub attention: AttentionBuffers,
    pub mlp: MlpBuffers,
    /// The last token's hidden state, `[1, hidden_size]`.
    pub last: Tensor<f32>,
    pub final_norm: Tensor<f32>,
    /// `[1, vocab_size]` on the device, then copied to `host_logits`.
    pub logits: Tensor<f32>,
    pub host_logits: Vec<f32>,
}

/// The tensors one layer writes around its attention and MLP, `[rows, hidden_size]`.
pub(crate) struct BlockBuffers {
    pub normalized: Tensor<f32>,
    pub attention_output: Tensor<f32>,
    pub x_after_attention: Tensor<f32>,
    pub mlp_output: Tensor<f32>,
}

/// The tensors one layer's attention writes.
pub(crate) struct AttentionBuffers {
    /// Projections before they are split into heads, `[rows, heads * head_dim]`.
    pub q_projected: Tensor<f32>,
    pub k_projected: Tensor<f32>,
    pub v_projected: Tensor<f32>,
    /// `[heads, rows, head_dim]`, or `[kv_heads, rows, head_dim]` for k and v.
    pub q: Tensor<f32>,
    pub k: Tensor<f32>,
    pub v: Tensor<f32>,
    pub q_normalized: Tensor<f32>,
    pub k_normalized: Tensor<f32>,
    pub q_rotated: Tensor<f32>,
    pub k_rotated: Tensor<f32>,
    /// `RoPE` tables for every position the cache can hold,
    /// `[capacity, head_dim / 2]`, built once instead of every pass.
    pub cos_table: Tensor<f32>,
    pub sin_table: Tensor<f32>,
    /// The table rows for this pass's positions, `[rows, head_dim / 2]`.
    pub cos: Tensor<f32>,
    pub sin: Tensor<f32>,
    /// `[heads, rows, capacity]`. Each pass uses the first columns, one for
    /// every token in the cache.
    pub scores: Tensor<f32>,
    pub probs: Tensor<f32>,
    /// The weighted sum of values, `[heads, rows, head_dim]`.
    pub mixed: Tensor<f32>,
    /// `mixed` with its heads merged back, `[rows, heads * head_dim]`.
    pub merged: Tensor<f32>,
}

/// The tensors one layer's MLP writes, `[rows, intermediate_size]`.
pub(crate) struct MlpBuffers {
    pub gate: Tensor<f32>,
    pub up: Tensor<f32>,
    pub gated: Tensor<f32>,
}

impl Workspace {
    /// Allocates every buffer for passes of `rows` new tokens against caches
    /// with room for `capacity` tokens.
    pub(crate) fn new(
        backend: &Backend,
        config: &Qwen3Config,
        rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        let hidden = [rows, config.hidden_size];
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let half = head_dim / 2;
        let wide = [rows, config.intermediate_size];

        let positions = (0..capacity)
            .map(|position| u32::try_from(position).expect("positions fit in u32"))
            .collect::<Vec<_>>();
        let (cos, sin) = rope_tables(&positions, head_dim, config.rope_theta);

        Ok(Self {
            rows,
            x: backend.zeros(&hidden)?,
            block: BlockBuffers {
                normalized: backend.zeros(&hidden)?,
                attention_output: backend.zeros(&hidden)?,
                x_after_attention: backend.zeros(&hidden)?,
                mlp_output: backend.zeros(&hidden)?,
            },
            attention: AttentionBuffers {
                q_projected: backend.zeros(&[rows, heads * head_dim])?,
                k_projected: backend.zeros(&[rows, kv_heads * head_dim])?,
                v_projected: backend.zeros(&[rows, kv_heads * head_dim])?,
                q: backend.zeros(&[heads, rows, head_dim])?,
                k: backend.zeros(&[kv_heads, rows, head_dim])?,
                v: backend.zeros(&[kv_heads, rows, head_dim])?,
                q_normalized: backend.zeros(&[heads, rows, head_dim])?,
                k_normalized: backend.zeros(&[kv_heads, rows, head_dim])?,
                q_rotated: backend.zeros(&[heads, rows, head_dim])?,
                k_rotated: backend.zeros(&[kv_heads, rows, head_dim])?,
                cos_table: backend.upload(&cos, &[capacity, half])?,
                sin_table: backend.upload(&sin, &[capacity, half])?,
                cos: backend.zeros(&[rows, half])?,
                sin: backend.zeros(&[rows, half])?,
                scores: backend.zeros(&[heads, rows, capacity])?,
                probs: backend.zeros(&[heads, rows, capacity])?,
                mixed: backend.zeros(&[heads, rows, head_dim])?,
                merged: backend.zeros(&[rows, heads * head_dim])?,
            },
            mlp: MlpBuffers {
                gate: backend.zeros(&wide)?,
                up: backend.zeros(&wide)?,
                gated: backend.zeros(&wide)?,
            },
            last: backend.zeros(&[1, config.hidden_size])?,
            final_norm: backend.zeros(&[1, config.hidden_size])?,
            logits: backend.zeros(&[1, config.vocab_size])?,
            host_logits: vec![0.0; config.vocab_size],
        })
    }

    /// Checks that a pass of `rows` tokens fits this workspace.
    pub(crate) fn check_rows(&self, rows: usize) -> Result<()> {
        if rows != self.rows {
            return Err(ModelError::WorkspaceRows {
                expected: self.rows,
                actual: rows,
            });
        }
        Ok(())
    }
}
