use std::path::Path;

use engine_core::SafeTensorCatalog;

use crate::{BackendError, check_rms_norm_shape};

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct RmsNormF32Case {
    pub name: &'static str,
    pub rows: usize,
    pub hidden_size: usize,
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
    pub expected: Vec<f32>,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct RmsNormBf16Case {
    pub name: &'static str,
    pub rows: usize,
    pub hidden_size: usize,
    pub input_bits: Vec<u16>,
    pub weight_bits: Vec<u16>,
    pub expected_bits: Vec<u16>,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct RmsNormFixtures {
    pub epsilon: f32,
    pub f32_cases: Vec<RmsNormF32Case>,
    pub bf16_cases: Vec<RmsNormBf16Case>,
}

pub(crate) fn load_fixtures(path: &Path) -> Result<RmsNormFixtures, BackendError> {
    let catalog = SafeTensorCatalog::open_files([path.to_path_buf()])?;
    let epsilon = load_epsilon(&catalog)?;

    Ok(RmsNormFixtures {
        epsilon,
        f32_cases: vec![
            load_f32_case(
                &catalog,
                "layer_input_2048_f32",
                "input",
                "weight",
                "output",
            )?,
            load_f32_case(
                &catalog,
                "layer_q_128_f32",
                "q_input",
                "q_weight",
                "q_output",
            )?,
            load_f32_case(
                &catalog,
                "qwen3_4b_width_2560_f32",
                "wide_input",
                "wide_weight",
                "wide_output",
            )?,
        ],
        bf16_cases: vec![load_bf16_case(
            &catalog,
            "layer_input_2048_bfloat16",
            "bf16_input",
            "bf16_weight",
            "bf16_output",
        )?],
    })
}

fn load_epsilon(catalog: &SafeTensorCatalog) -> Result<f32, BackendError> {
    let epsilon = catalog.load("eps")?;
    if !epsilon.descriptor.layout.shape.is_empty() {
        return Err(fixture_error(format!(
            "eps has shape {:?}, expected a scalar",
            epsilon.descriptor.layout.shape
        )));
    }
    let values = epsilon.to_f32_vec()?;
    let [epsilon] = values.as_slice() else {
        return Err(fixture_error("eps must contain exactly one value"));
    };
    Ok(*epsilon)
}

fn load_f32_case(
    catalog: &SafeTensorCatalog,
    name: &'static str,
    input_name: &str,
    weight_name: &str,
    output_name: &str,
) -> Result<RmsNormF32Case, BackendError> {
    let input = catalog.load(input_name)?;
    let weight = catalog.load(weight_name)?;
    let output = catalog.load(output_name)?;
    let (rows, hidden_size) = validate_shapes(name, &input, &weight, &output)?;

    Ok(RmsNormF32Case {
        name,
        rows,
        hidden_size,
        input: input.to_f32_vec()?,
        weight: weight.to_f32_vec()?,
        expected: output.to_f32_vec()?,
    })
}

fn load_bf16_case(
    catalog: &SafeTensorCatalog,
    name: &'static str,
    input_name: &str,
    weight_name: &str,
    output_name: &str,
) -> Result<RmsNormBf16Case, BackendError> {
    let input = catalog.load(input_name)?;
    let weight = catalog.load(weight_name)?;
    let output = catalog.load(output_name)?;
    let (rows, hidden_size) = validate_shapes(name, &input, &weight, &output)?;

    Ok(RmsNormBf16Case {
        name,
        rows,
        hidden_size,
        input_bits: input.to_bf16_bits()?,
        weight_bits: weight.to_bf16_bits()?,
        expected_bits: output.to_bf16_bits()?,
    })
}

fn validate_shapes(
    case: &str,
    input: &engine_core::OwnedTensor,
    weight: &engine_core::OwnedTensor,
    output: &engine_core::OwnedTensor,
) -> Result<(usize, usize), BackendError> {
    let Some(&hidden_size) = input.descriptor.layout.shape.last() else {
        return Err(fixture_error(format!(
            "{case} input must have at least one dimension"
        )));
    };
    let elements = input.descriptor.layout.element_count();
    if hidden_size == 0 || !elements.is_multiple_of(hidden_size) {
        return Err(fixture_error(format!("{case} input has an invalid shape")));
    }
    let rows = elements / hidden_size;
    check_rms_norm_shape(rows, hidden_size)?;

    if weight.descriptor.layout.shape != [hidden_size] {
        return Err(fixture_error(format!(
            "{case} weight has shape {:?}, expected [{hidden_size}]",
            weight.descriptor.layout.shape
        )));
    }
    if output.descriptor.layout.shape != input.descriptor.layout.shape {
        return Err(fixture_error(format!(
            "{case} output has shape {:?}, expected {:?}",
            output.descriptor.layout.shape, input.descriptor.layout.shape
        )));
    }
    Ok((rows, hidden_size))
}

fn fixture_error(message: impl Into<String>) -> BackendError {
    BackendError::RmsNormFixture(message.into())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rms_norm.safetensors")
    }

    #[test]
    fn loads_pytorch_rms_norm_cases() {
        let fixtures = load_fixtures(&fixture_path()).unwrap();

        assert_eq!(fixtures.epsilon, 1.0e-6);
        assert_eq!(fixtures.f32_cases.len(), 3);
        assert_eq!(fixtures.bf16_cases.len(), 1);
        assert_eq!(fixtures.f32_cases[0].hidden_size, 2048);
        assert_eq!(fixtures.f32_cases[1].hidden_size, 128);
        assert_eq!(fixtures.f32_cases[2].hidden_size, 2560);
        assert_eq!(fixtures.bf16_cases[0].hidden_size, 2048);
    }
}
