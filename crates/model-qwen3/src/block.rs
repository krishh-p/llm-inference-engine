use cuda_backend::{Backend, Tensor};

use crate::attention::Attention;
use crate::kv_cache::KvCache;
use crate::mlp::Mlp;
use crate::trace::Trace;
use crate::weights::LayerWeights;
use crate::workspace::Workspace;
use crate::{Qwen3Config, Result};

/// One transformer layer with its weights on the GPU.
pub(crate) struct Block<'a> {
    backend: &'a Backend,
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
            epsilon: config.rms_norm_eps,
            input_norm: backend.upload(&weights.input_norm, &[hidden_size])?,
            post_attention_norm: backend.upload(&weights.post_attention_norm, &[hidden_size])?,
            attention: Attention::new(backend, config, &weights.attention)?,
            mlp: Mlp::new(backend, config, &weights.mlp)?,
        })
    }

    /// Runs the new tokens in `workspace.x` through the layer: attention,
    /// then the MLP, each normalized first and added back to its input.
    ///
    /// `workspace.x` is `[rows, hidden_size]` and is updated in place.
    /// Prefill passes an empty cache and decode passes the cache from the
    /// tokens so far, and either way the new tokens are added to it.
    pub(crate) fn forward(
        &self,
        workspace: &mut Workspace,
        cache: &mut KvCache,
        trace: &mut Trace,
    ) -> Result<()> {
        let backend = self.backend;
        let Workspace {
            rows,
            x,
            block: b,
            attention,
            mlp,
            ..
        } = workspace;

        backend.rms_norm(&mut b.normalized, x, &self.input_norm, self.epsilon)?;
        trace.save(backend, "input_norm", &b.normalized)?;
        self.attention.forward(
            &b.normalized,
            &mut b.attention_output,
            *rows,
            cache,
            attention,
            trace,
        )?;

        // the residual add and the norm that follows it run as one kernel
        backend.add_rms_norm(
            &mut b.x_after_attention,
            &mut b.normalized,
            x,
            &b.attention_output,
            &self.post_attention_norm,
            self.epsilon,
        )?;
        trace.save(backend, "mlp_input", &b.normalized)?;
        self.mlp.forward(&b.normalized, &mut b.mlp_output, mlp)?;
        trace.save(backend, "mlp_output", &b.mlp_output)?;
        backend.add(x, &b.x_after_attention, &b.mlp_output)?;
        trace.save(backend, "output", x)?;
        Ok(())
    }
}
