use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;

use crate::{
    BufferOwner, DType, Device, EngineError, OwnedTensor, Result, TensorDescriptor, TensorLayout,
};

#[derive(Debug)]
pub struct SafeTensorCatalog {
    files: Vec<PathBuf>,
    tensors: BTreeMap<String, TensorDescriptor>,
}

impl SafeTensorCatalog {
    /// Opens every `.safetensors` file in a directory and indexes its tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be read, contains no
    /// `SafeTensors` files, or contains invalid or duplicate tensors.
    pub fn open_dir(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let entries = fs::read_dir(directory).map_err(|source| EngineError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        let mut files = Vec::new();
        for entry in entries {
            let path = entry
                .map_err(|source| EngineError::Io {
                    path: directory.to_path_buf(),
                    source,
                })?
                .path();
            if path
                .extension()
                .is_some_and(|extension| extension == "safetensors")
            {
                files.push(path);
            }
        }
        files.sort();
        if files.is_empty() {
            return Err(EngineError::NoSafeTensorFiles(directory.to_path_buf()));
        }
        Self::open_files(files)
    }

    /// Opens a set of `SafeTensors` files and builds one combined tensor index.
    ///
    /// # Errors
    ///
    /// Returns an error when a file cannot be read, is not valid `SafeTensors`,
    /// or defines a tensor name already present in another file.
    pub fn open_files(files: impl IntoIterator<Item = PathBuf>) -> Result<Self> {
        let mut catalog = Self {
            files: Vec::new(),
            tensors: BTreeMap::new(),
        };

        for path in files {
            let mapping = map_file(&path)?;
            let tensors =
                SafeTensors::deserialize(&mapping).map_err(|source| EngineError::SafeTensor {
                    path: path.clone(),
                    source,
                })?;

            for (name, view) in tensors.tensors() {
                let descriptor = TensorDescriptor {
                    name: name.clone(),
                    layout: TensorLayout::contiguous(
                        DType::from(view.dtype()),
                        view.shape().to_vec(),
                    ),
                    device: Device::Cpu,
                    owner: BufferOwner::ModelFile(path.clone()),
                    byte_len: view.data().len(),
                };
                if let Some(existing) = catalog.tensors.insert(name.clone(), descriptor) {
                    let BufferOwner::ModelFile(first) = existing.owner else {
                        unreachable!("catalog entries always come from model files")
                    };
                    return Err(EngineError::DuplicateTensor {
                        name,
                        first,
                        second: path,
                    });
                }
            }
            catalog.files.push(path);
        }

        if catalog.files.is_empty() {
            return Err(EngineError::NoSafeTensorFiles(PathBuf::from(
                "<empty list>",
            )));
        }
        Ok(catalog)
    }

    #[must_use]
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.tensors.values()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    /// Copies one indexed tensor into owned host memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor does not exist or its source file can
    /// no longer be read as valid `SafeTensors`.
    pub fn load(&self, name: &str) -> Result<OwnedTensor> {
        let descriptor = self
            .get(name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_owned()))?
            .clone();
        let BufferOwner::ModelFile(path) = &descriptor.owner else {
            unreachable!("catalog entries always come from model files")
        };
        let mapping = map_file(path)?;
        let tensors =
            SafeTensors::deserialize(&mapping).map_err(|source| EngineError::SafeTensor {
                path: path.clone(),
                source,
            })?;
        let view = tensors
            .tensor(name)
            .map_err(|source| EngineError::SafeTensor {
                path: path.clone(),
                source,
            })?;
        Ok(OwnedTensor {
            descriptor,
            bytes: view.data().to_vec(),
        })
    }
}

#[allow(unsafe_code)]
fn map_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(|source| EngineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: the mapping is read-only, and callers treat the mapped file as
    // immutable for the lifetime of the mapping.
    unsafe { MmapOptions::new().map(&file) }.map_err(|source| EngineError::Io {
        path: path.to_path_buf(),
        source,
    })
}
