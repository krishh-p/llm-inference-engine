use cuda_backend::{Backend, Tensor};

use crate::{ModelError, Result};

/// Host copies of the tensors a forward pass produced, saved by name so they
/// can be compared with the reference's trace.
#[derive(Default)]
pub(crate) struct Trace {
    tensors: Vec<(&'static str, Vec<f32>)>,
}

impl Trace {
    pub(crate) fn save(
        &mut self,
        backend: &Backend,
        name: &'static str,
        tensor: &Tensor<f32>,
    ) -> Result<()> {
        self.tensors.push((name, backend.download(tensor)?));
        Ok(())
    }

    pub(crate) fn get(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .iter()
            .find(|(saved, _)| *saved == name)
            .map(|(_, values)| values.as_slice())
            .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))
    }
}
