use cuda_backend::{Backend, Tensor};

use crate::attention::KvCache;
use crate::block::Block;
use crate::trace::Trace;
use crate::weights::{LayerWeights, load_tensor};
use crate::{ModelError, Qwen3Artifacts, Result};

/// The whole model, with all of its weights on the GPU.
pub(crate) struct Model<'a> {
    backend: &'a Backend,
    hidden_size: usize,
    vocab_size: usize,
    epsilon: f32,
    blocks: Vec<Block<'a>>,
    final_norm: Tensor<f32>,
    /// `[vocab_size, hidden_size]`, the same matrix as the embedding table.
    lm_head: Tensor<f32>,
}

impl<'a> Model<'a> {
    /// Uploads the weights of every layer, the final norm and the LM head.
    ///
    /// Each layer is widened to float32 on the host, uploaded, and dropped
    /// before the next one is read, so the host only holds one layer at a time.
    pub(crate) fn load(backend: &'a Backend, artifacts: &Qwen3Artifacts) -> Result<Self> {
        let config = &artifacts.config;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        if !config.tie_word_embeddings {
            return Err(ModelError::InvalidConfig(
                "tie_word_embeddings=false is not supported".to_owned(),
            ));
        }

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let weights = LayerWeights::load(artifacts, layer)?;
            blocks.push(Block::new(backend, config, &weights)?);
        }

        let final_norm = load_tensor(&artifacts.weights, "model.norm.weight", &[hidden_size])?;
        let lm_head = load_tensor(
            &artifacts.weights,
            "model.embed_tokens.weight",
            &[vocab_size, hidden_size],
        )?;

        Ok(Self {
            backend,
            hidden_size,
            vocab_size,
            epsilon: config.rms_norm_eps,
            blocks,
            final_norm: backend.upload(&final_norm, &[hidden_size])?,
            lm_head: backend.upload(&lm_head, &[vocab_size, hidden_size])?,
        })
    }

    /// Runs `rows` new tokens through every layer and predicts the next token.
    ///
    /// `caches` has one cache per layer from the tokens so far, and is empty
    /// for prefill. `x` is `[rows, hidden_size]`. The result is `[1, vocab_size]`
    /// logits and every layer's cache extended by the new tokens.
    ///
    /// The trace gets each layer's output as `layer_{i}_output`, then
    /// `final_norm` and `logits`.
    pub(crate) fn forward(
        &self,
        mut x: Tensor<f32>,
        rows: usize,
        caches: &[KvCache],
        trace: &mut Trace,
    ) -> Result<(Tensor<f32>, Vec<KvCache>)> {
        let backend = self.backend;
        let mut new_caches = Vec::with_capacity(self.blocks.len());

        for (layer, block) in self.blocks.iter().enumerate() {
            let (output, cache) = block.forward(&x, rows, caches.get(layer), &mut Trace::off())?;
            trace.save(backend, format!("layer_{layer}_output"), &output)?;
            x = output;
            new_caches.push(cache);
        }

        // only the last row predicts the next token
        let mut last = backend.zeros(&[1, self.hidden_size])?;
        backend.copy_rows(&mut last, 0, &x, rows - 1, 1)?;
        let mut normalized = backend.zeros(&[1, self.hidden_size])?;
        backend.rms_norm(&mut normalized, &last, &self.final_norm, self.epsilon)?;
        trace.save(backend, "final_norm", &normalized)?;

        let mut logits = backend.zeros(&[1, self.vocab_size])?;
        backend.matmul_transposed(&mut logits, &normalized, &self.lm_head)?;
        trace.save(backend, "logits", &logits)?;

        Ok((logits, new_caches))
    }
}
