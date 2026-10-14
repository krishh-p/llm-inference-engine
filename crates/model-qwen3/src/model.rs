use cuda_backend::{Backend, Tensor};

use crate::block::Block;
use crate::kv_cache::KvCache;
use crate::trace::Trace;
use crate::weights::{LayerWeights, load_tensor};
use crate::workspace::Workspace;
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

    /// Allocates an empty cache for every layer, each with room for
    /// `capacity` tokens.
    pub(crate) fn new_caches(&self, capacity: usize) -> Result<Vec<KvCache>> {
        (0..self.blocks.len())
            .map(|_| KvCache::new(self.backend, &self.config, capacity))
            .collect()
    }

    /// Allocates a workspace for passes of `rows` new tokens against caches
    /// with room for `capacity` tokens.
    pub(crate) fn new_workspace(&self, rows: usize, capacity: usize) -> Result<Workspace> {
        Workspace::new(self.backend, &self.config, rows, capacity)
    }

    /// Runs the whole prompt at once and fills the empty `caches` with it.
    ///
    /// The workspace must be sized for the prompt. The result is the logits
    /// that predict the first new token.
    pub(crate) fn prefill<'w>(
        &self,
        prompt_ids: &[u32],
        caches: &mut [KvCache],
        workspace: &'w mut Workspace,
        trace: &mut Trace,
    ) -> Result<&'w [f32]> {
        if let Some(cache) = caches.iter().find(|cache| cache.len > 0) {
            return Err(ModelError::CacheNotEmpty { len: cache.len });
        }
        self.forward(prompt_ids, caches, workspace, trace)
    }

    /// Runs one new token against everything in `caches` and adds it to them.
    ///
    /// The workspace must be sized for one token, and the same one is reused
    /// for every step. The result is the logits that predict the next token.
    pub(crate) fn decode_step<'w>(
        &self,
        token_id: u32,
        caches: &mut [KvCache],
        workspace: &'w mut Workspace,
        trace: &mut Trace,
    ) -> Result<&'w [f32]> {
        self.forward(&[token_id], caches, workspace, trace)
    }

    /// How many positions the model can attend over.
    pub(crate) fn max_positions(&self) -> usize {
        self.config.max_position_embeddings
    }

    /// Runs new tokens through every layer and copies the logits that
    /// predict the next token to the host.
    ///
    /// The trace gets `embedding`, each layer's output as `layer_{i}_output`,
    /// then `final_norm` and `logits`.
    fn forward<'w>(
        &self,
        token_ids: &[u32],
        caches: &mut [KvCache],
        workspace: &'w mut Workspace,
        trace: &mut Trace,
    ) -> Result<&'w [f32]> {
        let backend = self.backend;
        let rows = token_ids.len();
        self.config.check_token_ids(token_ids)?;
        self.config.check_cache_count(caches.len())?;
        workspace.check_rows(rows)?;
        // checked up front so a full cache fails before any layer has changed
        for cache in caches.iter() {
            cache.check_room(rows)?;
        }

        self.embed(token_ids, &mut workspace.x)?;
        trace.save(backend, "embedding", &workspace.x)?;

        for (layer, (block, cache)) in self.blocks.iter().zip(caches.iter_mut()).enumerate() {
            block.forward(workspace, cache, &mut Trace::off())?;
            trace.save_layer_output(backend, layer, &workspace.x)?;
        }

        // only the last row predicts the next token
        let ws = workspace;
        backend.copy_rows(&mut ws.last, 0, &ws.x, rows - 1, 1)?;
        backend.rms_norm(
            &mut ws.final_norm,
            &ws.last,
            &self.final_norm,
            self.config.rms_norm_eps,
        )?;
        trace.save(backend, "final_norm", &ws.final_norm)?;

        backend.matmul_transposed(&mut ws.logits, &ws.final_norm, &self.embed_tokens)?;
        trace.save(backend, "logits", &ws.logits)?;
        backend.download_into(&ws.logits, &mut ws.host_logits)?;
        Ok(&ws.host_logits)
    }

    /// Copies each token's row of the embedding table into `x`, `[tokens, hidden_size]`.
    fn embed(&self, token_ids: &[u32], x: &mut Tensor<f32>) -> Result<()> {
        for (row, &token_id) in token_ids.iter().enumerate() {
            let token_id = usize::try_from(token_id).expect("token IDs fit in usize");
            self.backend
                .copy_rows(x, row, &self.embed_tokens, token_id, 1)?;
        }
        Ok(())
    }
}
