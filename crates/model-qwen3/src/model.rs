use cuda_backend::{Backend, Tensor};

use crate::attention::KvCache;
use crate::block::Block;
use crate::trace::Trace;
use crate::weights::LayerWeights;
use crate::{Qwen3Artifacts, Result};

/// Every transformer layer of the model, with all of their weights on the GPU.
pub(crate) struct Model<'a> {
    backend: &'a Backend,
    blocks: Vec<Block<'a>>,
}

impl<'a> Model<'a> {
    /// Uploads the weights of every layer.
    ///
    /// Each layer is widened to float32 on the host, uploaded, and dropped
    /// before the next one is read, so the host only holds one layer at a time.
    pub(crate) fn load(backend: &'a Backend, artifacts: &Qwen3Artifacts) -> Result<Self> {
        let config = &artifacts.config;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let weights = LayerWeights::load(artifacts, layer)?;
            blocks.push(Block::new(backend, config, &weights)?);
        }

        Ok(Self { backend, blocks })
    }

    /// Runs `rows` new tokens through every layer in order.
    ///
    /// `caches` has one cache per layer from the tokens so far, and is empty
    /// for prefill. `x` is `[rows, hidden_size]`, and the result has the same
    /// shape and comes with every layer's cache extended by the new tokens.
    /// The trace gets each layer's output as `layer_{i}_output`.
    pub(crate) fn forward(
        &self,
        mut x: Tensor<f32>,
        rows: usize,
        caches: &[KvCache],
        trace: &mut Trace,
    ) -> Result<(Tensor<f32>, Vec<KvCache>)> {
        let mut new_caches = Vec::with_capacity(self.blocks.len());

        for (layer, block) in self.blocks.iter().enumerate() {
            let (output, cache) = block.forward(&x, rows, caches.get(layer), &mut Trace::off())?;
            trace.save(self.backend, format!("layer_{layer}_output"), &output)?;
            x = output;
            new_caches.push(cache);
        }

        Ok((x, new_caches))
    }
}
