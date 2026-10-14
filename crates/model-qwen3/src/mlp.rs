use cuda_backend::{Backend, Tensor};

use crate::weights::MlpWeights;
use crate::{Qwen3Config, Result};

/// One layer's MLP with its weights on the GPU.
pub(crate) struct Mlp<'a> {
    backend: &'a Backend,
    hidden_size: usize,
    intermediate_size: usize,
    gate_proj: Tensor<f32>,
    up_proj: Tensor<f32>,
    down_proj: Tensor<f32>,
}

impl<'a> Mlp<'a> {
    pub(crate) fn new(
        backend: &'a Backend,
        config: &Qwen3Config,
        weights: &MlpWeights,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Ok(Self {
            backend,
            hidden_size,
            intermediate_size,
            gate_proj: backend.upload(&weights.gate, &[intermediate_size, hidden_size])?,
            up_proj: backend.upload(&weights.up, &[intermediate_size, hidden_size])?,
            down_proj: backend.upload(&weights.down, &[hidden_size, intermediate_size])?,
        })
    }

    /// `(silu(x @ gate^T) * (x @ up^T)) @ down^T`, for `x` of `[rows, hidden_size]`.
    pub(crate) fn forward(&self, x: &Tensor<f32>, rows: usize) -> Result<Tensor<f32>> {
        let backend = self.backend;
        let wide = [rows, self.intermediate_size];

        let mut gate = backend.zeros(&wide)?;
        backend.matmul_transposed(&mut gate, x, &self.gate_proj)?;
        let mut up = backend.zeros(&wide)?;
        backend.matmul_transposed(&mut up, x, &self.up_proj)?;

        let mut gated = backend.zeros(&wide)?;
        backend.silu_mul(&mut gated, &gate, &up)?;

        let mut output = backend.zeros(&[rows, self.hidden_size])?;
        backend.matmul_transposed(&mut output, &gated, &self.down_proj)?;
        Ok(output)
    }
}
