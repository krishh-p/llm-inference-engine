use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse JSON from {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to read SafeTensors file {path}: {source}")]
    SafeTensor {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },

    #[error("no SafeTensors files were found in {0}")]
    NoSafeTensorFiles(PathBuf),

    #[error("tensor {name:?} appears in both {first} and {second}")]
    DuplicateTensor {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },

    #[error("tensor {0:?} was not found")]
    TensorNotFound(String),

    #[error("tensor {name:?} has dtype {actual}, expected {expected}")]
    TensorDtype {
        name: String,
        expected: &'static str,
        actual: String,
    },

    #[error("tensor {name:?} has {actual} data bytes, expected {expected}")]
    TensorDataLength {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("unsupported fixture manifest version {0}")]
    FixtureVersion(u32),

    #[error("fixture {path} has {actual} bytes, expected {expected}")]
    FixtureSize {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },

    #[error("fixture {path} has SHA-256 {actual}, expected {expected}")]
    FixtureChecksum {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("fixture {file} has no {name} tolerance in the manifest")]
    FixtureTolerance { file: String, name: String },

    #[error(
        "{name} differs from its fixture by up to {max_abs_error}, allowed {atol} + {rtol} * |expected|"
    )]
    FixtureMismatch {
        name: String,
        max_abs_error: f32,
        atol: f64,
        rtol: f64,
    },

    #[error("fixture {file} tensor {tensor:?}: {message}")]
    FixtureTensor {
        file: PathBuf,
        tensor: String,
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// Reads and deserializes a JSON file.
pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| EngineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| EngineError::Json {
        path: path.to_path_buf(),
        source,
    })
}
