use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

use crate::{ModelError, Result};

pub struct Qwen3Tokenizer {
    path: PathBuf,
    tokenizer: Tokenizer,
}

impl Qwen3Tokenizer {
    /// Loads a Hugging Face `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file is missing or cannot be parsed by the
    /// tokenizer library.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(ModelError::MissingArtifact(path.to_path_buf()));
        }
        let tokenizer = Tokenizer::from_file(path).map_err(|error| ModelError::Tokenizer {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            tokenizer,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Encodes one prompt without adding special tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer cannot encode the prompt.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(prompt, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ModelError::Tokenize {
                prompt: prompt.to_owned(),
                message: error.to_string(),
            })
    }
}
