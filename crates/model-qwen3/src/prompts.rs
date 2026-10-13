use std::path::Path;

use serde::Deserialize;

use crate::{ModelError, Qwen3Tokenizer, Result};

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct RegressionPromptSet {
    pub max_new_tokens: usize,
    pub cases: Vec<RegressionCase>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct RegressionCase {
    pub prompt: String,
    pub token_ids: Vec<u32>,
}

impl RegressionPromptSet {
    /// Loads the fixed tokenizer regression cases.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed, or when it has
    /// no test cases.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let set: Self = engine_core::read_json(path)?;
        if set.cases.is_empty() {
            return Err(ModelError::InvalidConfig(
                "regression prompt set is empty".to_owned(),
            ));
        }
        Ok(set)
    }

    /// Confirms that every prompt produces its recorded token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization fails or any result differs from the
    /// recorded IDs.
    pub fn validate_tokenizer(&self, tokenizer: &Qwen3Tokenizer) -> Result<()> {
        for case in &self.cases {
            let actual = tokenizer.encode(&case.prompt)?;
            if actual != case.token_ids {
                return Err(ModelError::TokenMismatch {
                    prompt: case.prompt.clone(),
                    expected: case.token_ids.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }
}
