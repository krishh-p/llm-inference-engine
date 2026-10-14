use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Engine(#[from] engine_core::EngineError),

    #[error(transparent)]
    Backend(#[from] cuda_backend::BackendError),

    #[error("invalid Qwen3 config: {0}")]
    InvalidConfig(String),

    #[error("required model artifact is missing: {0}")]
    MissingArtifact(PathBuf),

    #[error("required tensor {0:?} is missing")]
    MissingTensor(String),

    #[error("tensor {name:?} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("tensor {name:?} has dtype {actual}, expected {expected}")]
    DTypeMismatch {
        name: String,
        expected: String,
        actual: String,
    },

    #[error("failed to load tokenizer {path}: {message}")]
    Tokenizer { path: PathBuf, message: String },

    #[error("failed to tokenize prompt {prompt:?}: {message}")]
    Tokenize { prompt: String, message: String },

    #[error("failed to decode token IDs {token_ids:?}: {message}")]
    Detokenize {
        token_ids: Vec<u32>,
        message: String,
    },

    #[error("prompt {prompt:?} tokenized to {actual:?}, expected {expected:?}")]
    TokenMismatch {
        prompt: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },

    #[error("token ID {token_id} is outside the vocabulary of {vocab_size} tokens")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },

    #[error("forward pass got {actual} KV caches, expected {expected}")]
    CacheCount { expected: usize, actual: usize },

    #[error("the prompt has no tokens")]
    EmptyPrompt,

    #[error("max_new_tokens must be at least 1")]
    NoNewTokens,

    #[error(
        "{prompt_tokens} prompt tokens plus {max_new_tokens} new tokens do not fit in {max_positions} positions"
    )]
    ContextTooLong {
        prompt_tokens: usize,
        max_new_tokens: usize,
        max_positions: usize,
    },

    #[error("prompt {prompt:?} generated {actual:?}, expected {expected:?}")]
    GenerationMismatch {
        prompt: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },

    #[error("prompt {prompt:?} generated {cached:?} with caches but {uncached:?} without")]
    CacheMismatch {
        prompt: String,
        cached: Vec<u32>,
        uncached: Vec<u32>,
    },

    #[error("KV cache holds {len} of {capacity} tokens, so {rows} more do not fit")]
    CacheFull {
        len: usize,
        rows: usize,
        capacity: usize,
    },

    #[error("prefill needs empty KV caches, but one holds {len} tokens")]
    CacheNotEmpty { len: usize },

    #[error("workspace is sized for {expected} new tokens, but the pass has {actual}")]
    WorkspaceRows { expected: usize, actual: usize },

    #[error("check failed: {0}")]
    CheckFailed(String),

    #[error("fixture manifest field {0} is empty")]
    EmptyManifestField(&'static str),

    #[error("fixture revision {actual} does not match pinned model revision {expected}")]
    RevisionMismatch { expected: String, actual: String },
}

pub type Result<T> = std::result::Result<T, ModelError>;
