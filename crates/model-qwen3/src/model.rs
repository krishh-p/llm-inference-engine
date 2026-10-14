use cuda_backend::{Backend, Tensor};

use crate::attention::KvCache;
use crate::block::Block;
use crate::trace::Trace;
use crate::weights::{LayerWeights, load_tensor};
use crate::{ModelError, Qwen3Artifacts, Qwen3Config, Result};

/// The whole model, with all of its weights on the GPU.
pub(crate) struct Model<'a> {
    backend: &'a Backend,
    config: Qwen3Config,
    /// `[vocab_size, hidden_size]`. The model ties its LM head to this table,
    /// so the same matrix turns tokens into vectors and vectors into logits.
    embed_tokens: Tensor<f32>,
    blocks: Vec<Block<'a>>,
    final_norm: Tensor<f32>,
}

impl<'a> Model<'a> {
    /// Uploads the embedding table, the weights of every layer and the final norm.
    ///
    /// Each layer is widened to float32 on the host, uploaded, and dropped
    /// before the next one is read, so the host only holds one layer at a time.
    pub(crate) fn load(backend: &'a Backend, artifacts: &Qwen3Artifacts) -> Result<Self> {
        let config = &artifacts.config;
        let hidden_size = config.hidden_size;
        if !config.tie_word_embeddings {
            return Err(ModelError::InvalidConfig(
                "tie_word_embeddings=false is not supported".to_owned(),
            ));
        }

        let embed_tokens = load_tensor(
            &artifacts.weights,
            "model.embed_tokens.weight",
            &[config.vocab_size, hidden_size],
        )?;
        let embed_tokens = backend.upload(&embed_tokens, &[config.vocab_size, hidden_size])?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let weights = LayerWeights::load(artifacts, layer)?;
            blocks.push(Block::new(backend, config, &weights)?);
        }

        let final_norm = load_tensor(&artifacts.weights, "model.norm.weight", &[hidden_size])?;

        Ok(Self {
            backend,
            config: config.clone(),
            embed_tokens,
            blocks,
            final_norm: backend.upload(&final_norm, &[hidden_size])?,
        })
    }

    /// Runs new tokens through the model and predicts the next token.
    ///
    /// `caches` has one cache per layer from the tokens so far, and is empty
    /// for prefill. The result is `[1, vocab_size]` logits and every layer's
    /// cache extended by the new tokens.
    ///
    /// The trace gets `embedding`, each layer's output as `layer_{i}_output`,
    /// then `final_norm` and `logits`.
    pub(crate) fn forward(
        &self,
        token_ids: &[u32],
        caches: &[KvCache],
        trace: &mut Trace,
    ) -> Result<(Tensor<f32>, Vec<KvCache>)> {
        let backend = self.backend;
        let hidden_size = self.config.hidden_size;
        let rows = token_ids.len();
        self.config.check_token_ids(token_ids)?;
        self.config.check_cache_count(caches.len())?;

        let mut x = self.embed(token_ids)?;
        trace.save(backend, "embedding", &x)?;

        let mut new_caches = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (output, cache) = block.forward(&x, rows, caches.get(layer), &mut Trace::off())?;
            trace.save(backend, format!("layer_{layer}_output"), &output)?;
            x = output;
            new_caches.push(cache);
        }

        // only the last row predicts the next token
        let mut last = backend.zeros(&[1, hidden_size])?;
        backend.copy_rows(&mut last, 0, &x, rows - 1, 1)?;
        let mut normalized = backend.zeros(&[1, hidden_size])?;
        backend.rms_norm(
            &mut normalized,
            &last,
            &self.final_norm,
            self.config.rms_norm_eps,
        )?;
        trace.save(backend, "final_norm", &normalized)?;

        let mut logits = backend.zeros(&[1, self.config.vocab_size])?;
        backend.matmul_transposed(&mut logits, &normalized, &self.embed_tokens)?;
        trace.save(backend, "logits", &logits)?;

        Ok((logits, new_caches))
    }

    /// How many positions the model can attend over.
    pub(crate) fn max_positions(&self) -> usize {
        self.config.max_position_embeddings
    }

    /// Looks up each token's row of the embedding table, `[tokens, hidden_size]`.
    fn embed(&self, token_ids: &[u32]) -> Result<Tensor<f32>> {
        let mut x = self
            .backend
            .zeros(&[token_ids.len(), self.config.hidden_size])?;
        for (row, &token_id) in token_ids.iter().enumerate() {
            let token_id = usize::try_from(token_id).expect("token IDs fit in usize");
            self.backend
                .copy_rows(&mut x, row, &self.embed_tokens, token_id, 1)?;
        }
        Ok(x)
    }
}
