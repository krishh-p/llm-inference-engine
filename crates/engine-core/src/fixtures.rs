use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
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

/// How far a value may be from its fixture: `atol + rtol * |expected|`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Tolerance {
    #[must_use]
    pub fn allows(self, actual: f32, expected: f32) -> bool {
        // an infinite expected value would make the allowed error infinite too
        if !actual.is_finite() || !expected.is_finite() {
            return false;
        }

        let error = (f64::from(actual) - f64::from(expected)).abs();
        error <= self.atol + self.rtol * f64::from(expected).abs()
    }

    /// Returns the largest absolute error between `actual` and `expected`.
    ///
    /// # Errors
    ///
    /// Returns an error naming `name` when the lengths differ or any value is
    /// outside the tolerance.
    pub fn compare(self, name: &str, actual: &[f32], expected: &[f32]) -> Result<f32> {
        let mut max_abs_error = 0.0_f32;
        let mut within_tolerance = actual.len() == expected.len();

        for (&actual, &expected) in actual.iter().zip(expected) {
            let error = (actual - expected).abs();
            if !error.is_finite() {
                max_abs_error = f32::INFINITY;
                break;
            }
            max_abs_error = max_abs_error.max(error);
            within_tolerance &= self.allows(actual, expected);
        }

        if !within_tolerance || !max_abs_error.is_finite() {
            return Err(EngineError::FixtureMismatch {
                name: name.to_owned(),
                max_abs_error,
                atol: self.atol,
                rtol: self.rtol,
            });
        }
        Ok(max_abs_error)
    }
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

    /// Looks up the `{name}_atol` and `{name}_rtol` pair recorded for a file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file or either value is not in the manifest.
    pub fn tolerance(&self, file: &str, name: &str) -> Result<Tolerance> {
        let tolerances = self.files.get(file).map(|file| &file.tolerances);
        let value = |suffix: &str| {
            tolerances
                .and_then(|tolerances| tolerances.get(&format!("{name}_{suffix}")))
                .copied()
        };

        match (value("atol"), value("rtol")) {
            (Some(atol), Some(rtol)) => Ok(Tolerance { atol, rtol }),
            _ => Err(EngineError::FixtureTolerance {
                file: file.to_owned(),
                name: name.to_owned(),
            }),
        }
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

#[cfg(test)]
mod tests {
    use super::Tolerance;
    use crate::EngineError;

    const HALF: Tolerance = Tolerance {
        atol: 0.5,
        rtol: 0.0,
    };

    #[test]
    fn compare_returns_the_largest_error() {
        let max_abs_error = HALF.compare("case", &[1.0, 2.25, 3.0], &[1.0, 2.0, 3.5]);

        assert_eq!(max_abs_error.unwrap(), 0.5);
    }

    #[test]
    fn compare_rejects_a_value_outside_the_tolerance() {
        let error = HALF
            .compare("prefill_q", &[1.0, 3.0], &[1.0, 2.0])
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "prefill_q differs from its fixture by up to 1, allowed 0.5 + 0 * |expected|"
        );
    }

    #[test]
    fn compare_rejects_nan_and_missing_values() {
        assert!(matches!(
            HALF.compare("case", &[f32::NAN], &[1.0]),
            Err(EngineError::FixtureMismatch { .. })
        ));
        assert!(matches!(
            HALF.compare("case", &[1.0], &[1.0, 2.0]),
            Err(EngineError::FixtureMismatch { .. })
        ));
    }

    #[test]
    fn tolerance_grows_with_the_expected_value() {
        let tolerance = Tolerance {
            atol: 0.5,
            rtol: 0.25,
        };

        // allowed error is 0.5 + 0.25 * 8 = 2.5
        assert!(tolerance.allows(10.5, 8.0));
        assert!(tolerance.allows(5.5, 8.0));
        assert!(!tolerance.allows(10.75, 8.0));
        assert!(tolerance.allows(-10.5, -8.0));
    }

    #[test]
    fn tolerance_rejects_values_that_are_not_finite() {
        let tolerance = Tolerance {
            atol: 1.0,
            rtol: 1.0,
        };

        assert!(!tolerance.allows(f32::NAN, 1.0));
        assert!(!tolerance.allows(f32::INFINITY, 1.0));
        assert!(!tolerance.allows(1.0, f32::INFINITY));
        assert!(!tolerance.allows(1.0, f32::NAN));
        assert!(!tolerance.allows(f32::INFINITY, f32::INFINITY));
    }
}
