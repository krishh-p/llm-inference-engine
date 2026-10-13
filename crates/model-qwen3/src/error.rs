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

    #[error("prompt {prompt:?} tokenized to {actual:?}, expected {expected:?}")]
    TokenMismatch {
        prompt: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },

    #[error("fixture revision {actual} does not match pinned model revision {expected}")]
    RevisionMismatch { expected: String, actual: String },

    #[error(
        "{name} differs from its fixture by up to {max_abs_error}, allowed {atol} + {rtol} * |expected|"
    )]
    FixtureMismatch {
        name: String,
        max_abs_error: f32,
        atol: f64,
        rtol: f64,
    },
}

pub type Result<T> = std::result::Result<T, ModelError>;
