use cuda_backend::{Backend, Tensor};

use crate::attention::Attention;
use crate::kv_cache::KvCache;
use crate::mlp::Mlp;
use crate::trace::Trace;
use crate::weights::LayerWeights;
use crate::{Qwen3Config, Result};

/// One transformer layer with its weights on the GPU.
pub(crate) struct Block<'a> {
    backend: &'a Backend,
    hidden_size: usize,
    epsilon: f32,
    input_norm: Tensor<f32>,
    post_attention_norm: Tensor<f32>,
    attention: Attention<'a>,
    mlp: Mlp<'a>,
}

impl<'a> Block<'a> {
    pub(crate) fn new(
        backend: &'a Backend,
        config: &Qwen3Config,
        weights: &LayerWeights,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;

        Ok(Self {
            backend,
            hidden_size,
            epsilon: config.rms_norm_eps,
            input_norm: backend.upload(&weights.input_norm, &[hidden_size])?,
            post_attention_norm: backend.upload(&weights.post_attention_norm, &[hidden_size])?,
            attention: Attention::new(backend, config, &weights.attention)?,
            mlp: Mlp::new(backend, config, &weights.mlp)?,
        })
    }

    /// Runs `rows` new tokens through the layer: attention, then the MLP,
    /// each normalized first and added back to its input.
    ///
    /// Prefill passes an empty cache and decode passes the cache from the
    /// tokens so far, and either way the new tokens are added to it. `x` is
    /// `[rows, hidden_size]`, and the result has the same shape.
    pub(crate) fn forward(
        &self,
        x: &Tensor<f32>,
        rows: usize,
        cache: &mut KvCache,
        trace: &mut Trace,
    ) -> Result<Tensor<f32>> {
        let backend = self.backend;
        let shape = [rows, self.hidden_size];

        let mut normalized = backend.zeros(&shape)?;
        backend.rms_norm(&mut normalized, x, &self.input_norm, self.epsilon)?;
        trace.save(backend, "input_norm", &normalized)?;
        let attention_output = self.attention.forward(&normalized, rows, cache, trace)?;

        // the residual add and the norm that follows it run as one kernel
        let mut x_after_attention = backend.zeros(&shape)?;
        let mut normalized = backend.zeros(&shape)?;
        backend.add_rms_norm(
            &mut x_after_attention,
            &mut normalized,
            x,
            &attention_output,
            &self.post_attention_norm,
            self.epsilon,
        )?;
        trace.save(backend, "mlp_input", &normalized)?;
        let mlp_output = self.mlp.forward(&normalized, rows)?;
        trace.save(backend, "mlp_output", &mlp_output)?;
        let mut output = backend.zeros(&shape)?;
        backend.add(&mut output, &x_after_attention, &mlp_output)?;
        trace.save(backend, "output", &output)?;
        Ok(output)
    }
}
