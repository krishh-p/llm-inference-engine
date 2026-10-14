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

    pub(crate) fn get(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .iter()
            .find(|(saved, _)| saved == name)
            .map(|(_, values)| values.as_slice())
            .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))
    }
}
