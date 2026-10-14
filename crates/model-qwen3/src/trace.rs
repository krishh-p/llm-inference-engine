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

    /// Saves one layer's output as `layer_{layer}_output`. The name is only
    /// built while recording, so a decode step with tracing off allocates
    /// nothing here.
    pub(crate) fn save_layer_output(
        &mut self,
        backend: &Backend,
        layer: usize,
        tensor: &Tensor<f32>,
    ) -> Result<()> {
        if !self.recording {
            return Ok(());
        }
        self.save(backend, format!("layer_{layer}_output"), tensor)
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

    /// Saves the first `cols` columns of every row of a `[heads, rows, width]`
    /// tensor, which are the columns in use when it is sized for a whole cache.
    pub(crate) fn save_columns(
        &mut self,
        backend: &Backend,
        name: impl Into<String>,
        tensor: &Tensor<f32>,
        cols: usize,
    ) -> Result<()> {
        if !self.recording {
            return Ok(());
        }
        let width =
            usize::try_from(tensor.shape()[2]).expect("tensor dimensions are never negative");

        let values = backend.download(tensor)?;
        let used = values
            .chunks(width)
            .flat_map(|row| &row[..cols])
            .copied()
            .collect();
        self.tensors.push((name.into(), used));
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
