use std::path::Path;

use serde::Deserialize;

use crate::{ModelError, Result};

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Qwen3Config {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

impl Qwen3Config {
    /// Loads and validates a Qwen3 `config.json` file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed, or when the
    /// configuration is incompatible with this implementation.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = engine_core::read_json(path)?;
        config.validate()?;
        Ok(config)
    }

    /// Checks the model assumptions needed by the Qwen3 implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture or tensor dimensions are not a
    /// supported Qwen3 configuration.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3" {
            return Err(ModelError::InvalidConfig(format!(
                "model_type is {:?}, expected \"qwen3\"",
                self.model_type
            )));
        }
        if !self
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3ForCausalLM")
        {
            return Err(ModelError::InvalidConfig(
                "architectures does not contain Qwen3ForCausalLM".to_owned(),
            ));
        }
        if self.hidden_size != self.num_attention_heads * self.head_dim {
            return Err(ModelError::InvalidConfig(format!(
                "hidden_size {} does not equal num_attention_heads {} * head_dim {}",
                self.hidden_size, self.num_attention_heads, self.head_dim
            )));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ModelError::InvalidConfig(format!(
                "num_attention_heads {} must be divisible by num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
        {
            return Err(ModelError::InvalidConfig(
                "layer count and tensor dimensions must be non-zero".to_owned(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(ModelError::InvalidConfig(format!(
                "hidden_act is {:?}, expected \"silu\"",
                self.hidden_act
            )));
        }
        if self.attention_bias {
            return Err(ModelError::InvalidConfig(
                "attention_bias=true is not supported".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3Config;

    #[test]
    fn rejects_inconsistent_hidden_size() {
        let mut config: Qwen3Config =
            serde_json::from_str(include_str!("../../../tests/model-config.json")).unwrap();
        config.hidden_size = 2000;
        let error = config.validate().unwrap_err();
        assert!(error.to_string().contains("hidden_size"));
    }

    #[test]
    fn pinned_config_matches_reference_shapes() {
        let config: Qwen3Config =
            serde_json::from_str(include_str!("../../../tests/model-config.json")).unwrap();
        config.validate().unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.vocab_size, 151_936);
    }
}
