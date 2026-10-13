use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{EngineError, Result, SafeTensorCatalog, read_json};

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureManifest {
    pub format_version: u32,
    pub model: FixtureModel,
    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub expected_greedy_token_ids: Vec<u32>,
    pub max_logit_diff_vs_hugging_face: f64,
    pub max_logit_diff_tolerance: f64,
    pub files: BTreeMap<String, FixtureFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureModel {
    pub id: String,
    pub revision: String,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureFile {
    pub tensors: BTreeMap<String, FixtureTensor>,
    #[serde(default)]
    pub tolerances: BTreeMap<String, f64>,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureTensor {
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureValidation {
    pub file_count: usize,
    pub tensor_count: usize,
    pub total_bytes: u64,
}

impl FixtureManifest {
    /// Loads a version 1 fixture manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, contains invalid JSON, or
    /// uses an unsupported manifest version.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let manifest: Self = read_json(path)?;
        if manifest.format_version != 1 {
            return Err(EngineError::FixtureVersion(manifest.format_version));
        }
        Ok(manifest)
    }

    /// Checks every fixture's size, checksum, and tensor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixture is missing or differs from the
    /// information recorded in the manifest.
    pub fn validate_files(&self, directory: impl AsRef<Path>) -> Result<FixtureValidation> {
        let directory = directory.as_ref();
        let mut tensor_count = 0;
        let mut total_bytes = 0;

        for (filename, expected_file) in &self.files {
            let path = directory.join(filename);
            let bytes = fs::read(&path).map_err(|source| EngineError::Io {
                path: path.clone(),
                source,
            })?;
            let actual_size = bytes.len() as u64;
            if actual_size != expected_file.bytes {
                return Err(EngineError::FixtureSize {
                    path,
                    expected: expected_file.bytes,
                    actual: actual_size,
                });
            }

            let actual_hash = lower_hex(&Sha256::digest(&bytes));
            if actual_hash != expected_file.sha256 {
                return Err(EngineError::FixtureChecksum {
                    path,
                    expected: expected_file.sha256.clone(),
                    actual: actual_hash,
                });
            }

            let catalog = SafeTensorCatalog::open_files([path.clone()])?;
            if catalog.len() != expected_file.tensors.len() {
                return Err(EngineError::FixtureTensor {
                    file: path,
                    tensor: "<catalog>".to_owned(),
                    message: format!(
                        "contains {} tensors, expected {}",
                        catalog.len(),
                        expected_file.tensors.len()
                    ),
                });
            }

            for (name, expected_tensor) in &expected_file.tensors {
                let actual = catalog
                    .get(name)
                    .ok_or_else(|| EngineError::FixtureTensor {
                        file: path.clone(),
                        tensor: name.clone(),
                        message: "is missing".to_owned(),
                    })?;
                if actual.layout.shape != expected_tensor.shape {
                    return Err(EngineError::FixtureTensor {
                        file: path.clone(),
                        tensor: name.clone(),
                        message: format!(
                            "shape {:?}, expected {:?}",
                            actual.layout.shape, expected_tensor.shape
                        ),
                    });
                }
                if actual.layout.dtype.as_str() != expected_tensor.dtype {
                    return Err(EngineError::FixtureTensor {
                        file: path.clone(),
                        tensor: name.clone(),
                        message: format!(
                            "dtype {}, expected {}",
                            actual.layout.dtype, expected_tensor.dtype
                        ),
                    });
                }
            }

            tensor_count += catalog.len();
            total_bytes += actual_size;
        }

        Ok(FixtureValidation {
            file_count: self.files.len(),
            tensor_count,
            total_bytes,
        })
    }
}

fn lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}
