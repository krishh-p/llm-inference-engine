use std::path::Path;

use engine_core::SafeTensorCatalog;

use crate::{BackendError, check_rms_norm_shape};

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct RmsNormF32Case {
    pub rows: usize,
    pub hidden_size: usize,
    pub input: Vec<f32>,
    pub weight: Vec<f32>,
    pub epsilon: f32,
    pub expected: Vec<f32>,
}

pub(crate) fn load_fixture(path: &Path) -> Result<RmsNormF32Case, BackendError> {
    let catalog = SafeTensorCatalog::open_files([path.to_path_buf()])?;
    let input = catalog.load("input")?;
    let weight = catalog.load("weight")?;
    let epsilon = catalog.load("eps")?;
    let expected = catalog.load("output")?;

    let [rows, hidden_size] = input.descriptor.layout.shape.as_slice() else {
        return Err(fixture_error("input must have shape [rows, hidden_size]"));
    };
    let (rows, hidden_size) = (*rows, *hidden_size);
    check_rms_norm_shape(rows, hidden_size)?;

    if weight.descriptor.layout.shape != [hidden_size] {
        return Err(fixture_error(format!(
            "weight has shape {:?}, expected [{hidden_size}]",
            weight.descriptor.layout.shape
        )));
    }
    if expected.descriptor.layout.shape != [rows, hidden_size] {
        return Err(fixture_error(format!(
            "output has shape {:?}, expected [{rows}, {hidden_size}]",
            expected.descriptor.layout.shape
        )));
    }
    if !epsilon.descriptor.layout.shape.is_empty() {
        return Err(fixture_error(format!(
            "eps has shape {:?}, expected a scalar",
            epsilon.descriptor.layout.shape
        )));
    }

    let epsilon_values = epsilon.to_f32_vec()?;
    let [epsilon] = epsilon_values.as_slice() else {
        return Err(fixture_error("eps must contain exactly one value"));
    };

    Ok(RmsNormF32Case {
        rows,
        hidden_size,
        input: input.to_f32_vec()?,
        weight: weight.to_f32_vec()?,
        epsilon: *epsilon,
        expected: expected.to_f32_vec()?,
    })
}

#[cfg(any(target_os = "linux", test))]
pub(crate) fn reference_f32(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
) -> Vec<f32> {
    let hidden_size_f32 =
        f32::from(u16::try_from(hidden_size).expect("validated RMSNorm hidden size fits in u16"));
    let mut output = Vec::with_capacity(input.len());

    for row in input.chunks_exact(hidden_size).take(rows) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / hidden_size_f32;
        let inverse_rms = (mean_square + epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(value, scale)| value * inverse_rms * scale),
        );
    }

    output
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
    fn loads_qwen3_fixture() {
        let fixture = load_fixture(&fixture_path()).unwrap();

        assert_eq!(fixture.rows, 5);
        assert_eq!(fixture.hidden_size, 2048);
        assert_eq!(fixture.input.len(), 5 * 2048);
        assert_eq!(fixture.weight.len(), 2048);
        assert_eq!(fixture.expected.len(), 5 * 2048);
        assert_eq!(fixture.epsilon, 1.0e-6);
    }

    #[test]
    fn reference_normalizes_each_row() {
        let output = reference_f32(&[3.0, 4.0, 0.0, 5.0], &[1.0, 1.0], 2, 2, 0.0);
        let root_two = 2.0_f32.sqrt();
        let expected = [0.6 * root_two, 0.8 * root_two, 0.0, root_two];

        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6)
        );
    }
}
