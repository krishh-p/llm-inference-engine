use cuda_backend::{Backend, Tensor};

use crate::{ModelError, Result};

/// Host copies of the tensors a forward pass produced, saved by name so they
/// can be compared with the reference's trace.
///
/// A trace made with [`Trace::off`] saves nothing, so a normal run does not
/// copy its tensors back to the host.
pub(crate) struct Trace {
    recording: bool,
    tensors: Vec<(String, Vec<f32>)>,
}

impl Trace {
    pub(crate) fn recording() -> Self {
        Self {
            recording: true,
            tensors: Vec::new(),
        }
    }

    pub(crate) fn off() -> Self {
        Self {
            recording: false,
            tensors: Vec::new(),
        }
    }

    pub(crate) fn save(
        &mut self,
        backend: &Backend,
        name: impl Into<String>,
        tensor: &Tensor<f32>,
    ) -> Result<()> {
        if self.recording {
            self.tensors.push((name.into(), backend.download(tensor)?));
        }
        Ok(())
    }

    /// Saves the first `rows` rows of every head of a `[heads, capacity, width]`
    /// tensor, which are the filled rows of a KV cache.
    pub(crate) fn save_rows(
        &mut self,
        backend: &Backend,
        name: impl Into<String>,
        tensor: &Tensor<f32>,
        rows: usize,
    ) -> Result<()> {
        if !self.recording {
            return Ok(());
        }
        let shape = tensor.shape();
        let head =
            usize::try_from(shape[1] * shape[2]).expect("tensor dimensions are never negative");
        let width = usize::try_from(shape[2]).expect("tensor dimensions are never negative");

        let values = backend.download(tensor)?;
        let filled = values
            .chunks(head)
            .flat_map(|head| &head[..rows * width])
            .copied()
            .collect();
        self.tensors.push((name.into(), filled));
        Ok(())
    }

    pub(crate) fn get(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .iter()
            .find(|(saved, _)| saved == name)
            .map(|(_, values)| values.as_slice())
            .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))
    }
}
