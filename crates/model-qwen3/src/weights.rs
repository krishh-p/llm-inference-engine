use engine_core::SafeTensorCatalog;

use crate::{ModelError, Qwen3Artifacts, Result};

/// The attention weights of one layer, widened to float32 on the host.
///
/// Each projection is row-major `[output_width, input_width]`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct AttentionWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

impl AttentionWeights {
    pub(crate) fn load(artifacts: &Qwen3Artifacts, layer: usize) -> Result<Self> {
        let config = &artifacts.config;
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let q_width = config.num_attention_heads * head_dim;
        let kv_width = config.num_key_value_heads * head_dim;

        let prefix = format!("model.layers.{layer}.self_attn.");
        let load = |name: &str, shape: &[usize]| {
            load_tensor(&artifacts.weights, &format!("{prefix}{name}.weight"), shape)
        };

        Ok(Self {
            q_proj: load("q_proj", &[q_width, hidden_size])?,
            k_proj: load("k_proj", &[kv_width, hidden_size])?,
            v_proj: load("v_proj", &[kv_width, hidden_size])?,
            o_proj: load("o_proj", &[hidden_size, q_width])?,
            q_norm: load("q_norm", &[head_dim])?,
            k_norm: load("k_norm", &[head_dim])?,
        })
    }
}

/// Loads a tensor as float32 after checking its shape.
pub(crate) fn load_tensor(
    catalog: &SafeTensorCatalog,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let tensor = catalog.load(name)?;
    if tensor.descriptor.layout.shape != shape {
        return Err(ModelError::ShapeMismatch {
            name: name.to_owned(),
            expected: shape.to_vec(),
            actual: tensor.descriptor.layout.shape,
        });
    }
    Ok(tensor.widen_to_f32_vec()?)
}
