use cuda_backend::{Backend, Tensor};

use crate::weights::MlpWeights;
use crate::workspace::MlpBuffers;
use crate::{Qwen3Config, Result};

/// One layer's MLP with its weights on the GPU.
pub(crate) struct Mlp<'a> {
    backend: &'a Backend,
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
            gate_proj: backend.upload(&weights.gate, &[intermediate_size, hidden_size])?,
            up_proj: backend.upload(&weights.up, &[intermediate_size, hidden_size])?,
            down_proj: backend.upload(&weights.down, &[hidden_size, intermediate_size])?,
        })
    }

    /// Writes `(silu(x @ gate^T) * (x @ up^T)) @ down^T` to `output`, for `x`
    /// of `[rows, hidden_size]`.
    pub(crate) fn forward(
        &self,
        x: &Tensor<f32>,
        output: &mut Tensor<f32>,
        buffers: &mut MlpBuffers,
    ) -> Result<()> {
        let backend = self.backend;
        backend.matmul_transposed(&mut buffers.gate, x, &self.gate_proj)?;
        backend.matmul_transposed(&mut buffers.up, x, &self.up_proj)?;
        backend.silu_mul(&mut buffers.gated, &buffers.gate, &buffers.up)?;
        backend.matmul_transposed(output, &buffers.gated, &self.down_proj)?;
        Ok(())
    }
}
