use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use engine_core::{DType, SafeTensorCatalog, TensorDescriptor};

use crate::{ModelError, Qwen3Config, Qwen3Tokenizer, Result};

pub struct Qwen3Artifacts {
    pub directory: PathBuf,
    pub config: Qwen3Config,
    pub tokenizer: Qwen3Tokenizer,
    pub weights: SafeTensorCatalog,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSummary {
    pub architecture: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub vocabulary_size: usize,
    pub max_sequence_length: usize,
    pub weight_files: usize,
    pub tensors: usize,
    pub stored_parameters: u64,
    pub dtypes: Vec<String>,
}

impl Qwen3Artifacts {
    /// Opens and validates a local Qwen3 model snapshot.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let config_path = directory.join("config.json");
        let tokenizer_path = directory.join("tokenizer.json");
        if !config_path.is_file() {
            return Err(ModelError::MissingArtifact(config_path));
        }
        if !tokenizer_path.is_file() {
            return Err(ModelError::MissingArtifact(tokenizer_path));
        }

        let artifacts = Self {
            directory: directory.to_path_buf(),
            config: Qwen3Config::load(config_path)?,
            tokenizer: Qwen3Tokenizer::load(tokenizer_path)?,
            weights: SafeTensorCatalog::open_dir(directory)?,
        };
        artifacts.validate_weights()?;
        Ok(artifacts)
    }

    #[must_use]
    pub fn summary(&self) -> ModelSummary {
        let dtypes = self
            .weights
            .tensors()
            .map(|tensor| tensor.layout.dtype.to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let stored_parameters = self
            .weights
            .tensors()
            .map(|tensor| u64::try_from(tensor.layout.element_count()).unwrap_or(u64::MAX))
            .sum();
        ModelSummary {
            architecture: "Qwen3ForCausalLM".to_owned(),
            hidden_size: self.config.hidden_size,
            intermediate_size: self.config.intermediate_size,
            layers: self.config.num_hidden_layers,
            attention_heads: self.config.num_attention_heads,
            key_value_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim,
            vocabulary_size: self.config.vocab_size,
            max_sequence_length: self.config.max_position_embeddings,
            weight_files: self.weights.files().len(),
            tensors: self.weights.len(),
            stored_parameters,
            dtypes,
        }
    }

    /// Checks that every required weight has the configured shape and dtype.
    pub fn validate_weights(&self) -> Result<()> {
        let config = &self.config;
        let dtype = model_dtype(&config.torch_dtype)?;
        require(
            &self.weights,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
            &dtype,
        )?;
        require(
            &self.weights,
            "model.norm.weight",
            &[config.hidden_size],
            &dtype,
        )?;

        if config.tie_word_embeddings {
            if let Some(tensor) = self.weights.get("lm_head.weight") {
                validate_tensor(tensor, &[config.vocab_size, config.hidden_size], &dtype)?;
            }
        } else {
            require(
                &self.weights,
                "lm_head.weight",
                &[config.vocab_size, config.hidden_size],
                &dtype,
            )?;
        }

        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.");
            require(
                &self.weights,
                &format!("{prefix}input_layernorm.weight"),
                &[config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}post_attention_layernorm.weight"),
                &[config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}self_attn.q_proj.weight"),
                &[q_width, config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}self_attn.k_proj.weight"),
                &[kv_width, config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}self_attn.v_proj.weight"),
                &[kv_width, config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}self_attn.o_proj.weight"),
                &[config.hidden_size, q_width],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}self_attn.q_norm.weight"),
                &[config.head_dim],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}self_attn.k_norm.weight"),
                &[config.head_dim],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}mlp.gate_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}mlp.up_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
                &dtype,
            )?;
            require(
                &self.weights,
                &format!("{prefix}mlp.down_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
                &dtype,
            )?;
        }
        Ok(())
    }
}

fn model_dtype(name: &str) -> Result<DType> {
    match name {
        "bfloat16" => Ok(DType::BFloat16),
        "float16" => Ok(DType::Float16),
        "float32" => Ok(DType::Float32),
        other => Err(ModelError::InvalidConfig(format!(
            "unsupported torch_dtype {other:?}"
        ))),
    }
}

fn require(catalog: &SafeTensorCatalog, name: &str, shape: &[usize], dtype: &DType) -> Result<()> {
    let tensor = catalog
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    validate_tensor(tensor, shape, dtype)
}

fn validate_tensor(tensor: &TensorDescriptor, shape: &[usize], dtype: &DType) -> Result<()> {
    if tensor.layout.shape != shape {
        return Err(ModelError::ShapeMismatch {
            name: tensor.name.clone(),
            expected: shape.to_vec(),
            actual: tensor.layout.shape.clone(),
        });
    }
    if &tensor.layout.dtype != dtype {
        return Err(ModelError::DTypeMismatch {
            name: tensor.name.clone(),
            expected: dtype.to_string(),
            actual: tensor.layout.dtype.to_string(),
        });
    }
    Ok(())
}
