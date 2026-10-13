use std::path::Path;

use engine_core::{FixtureManifest, SafeTensorCatalog, Tolerance};

use crate::{BackendError, RopeTableResult};

const FIXTURE_FILE: &str = "rope.safetensors";

/// One tensor to rotate, laid out as `[heads, rows, head_dim]`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct RopeCase {
    pub name: &'static str,
    pub heads: usize,
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct RopeFixtures {
    pub positions: Vec<u32>,
    pub head_dim: usize,
    pub theta: f32,
    /// The reference tables, each `[rows, head_dim / 2]`.
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub cases: Vec<RopeCase>,
    pub rotation_tolerance: Tolerance,
    pub table_tolerance: Tolerance,
}

/// Builds the cos and sin tables, each `[positions, head_dim / 2]` in row-major order.
///
/// Column `i` turns at frequency `theta^(-i / half)`, so the first columns
/// turn quickly and the last ones slowly. The arithmetic is float32 to follow
/// the Python reference.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
// Positions and column indexes are far below 2^24, so float32 holds them exactly.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn rope_tables(positions: &[u32], head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);

    for &position in positions {
        for index in 0..half {
            let frequency = 1.0 / theta.powf(index as f32 / half as f32);
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }

    (cos, sin)
}

/// Compares the tables generated here with the ones the reference recorded.
///
/// # Errors
///
/// Returns an error when either table is outside the fixture tolerance.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_tables(fixtures: &RopeFixtures) -> Result<RopeTableResult, BackendError> {
    let (cos, sin) = rope_tables(&fixtures.positions, fixtures.head_dim, fixtures.theta);
    let tolerance = fixtures.table_tolerance;

    Ok(RopeTableResult {
        positions: fixtures.positions.clone(),
        cos_max_abs_error: tolerance.compare("rope_cos_table", &cos, &fixtures.cos)?,
        sin_max_abs_error: tolerance.compare("rope_sin_table", &sin, &fixtures.sin)?,
        tolerance,
        passed: true,
    })
}

pub(crate) fn load_fixtures(directory: &Path) -> Result<RopeFixtures, BackendError> {
    let manifest = FixtureManifest::load(directory.join("manifest.json"))?;
    let catalog = SafeTensorCatalog::open_files([directory.join(FIXTURE_FILE)])?;

    let positions = catalog
        .load("positions")?
        .to_i64_vec()?
        .into_iter()
        .map(|position| {
            u32::try_from(position)
                .map_err(|_| fixture_error(format!("position {position} is out of range")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rows = positions.len();

    let cos = catalog.load("cos")?;
    let &[table_rows, half] = cos.descriptor.layout.shape.as_slice() else {
        return Err(fixture_error("cos must have two dimensions"));
    };
    if table_rows != rows || half == 0 {
        return Err(fixture_error(format!(
            "cos has shape {:?}, expected one row for each of {rows} positions",
            cos.descriptor.layout.shape
        )));
    }
    let head_dim = half * 2;

    let theta = catalog.load("theta")?.to_f32_vec()?;
    let [theta] = theta.as_slice() else {
        return Err(fixture_error("theta must contain exactly one value"));
    };

    Ok(RopeFixtures {
        positions,
        head_dim,
        theta: *theta,
        cos: cos.to_f32_vec()?,
        sin: load_tensor(&catalog, "sin", &[rows, half])?,
        cases: vec![
            load_case(&catalog, "rope_q", "q_input", "q_output", rows, head_dim)?,
            load_case(&catalog, "rope_k", "k_input", "k_output", rows, head_dim)?,
        ],
        rotation_tolerance: manifest.tolerance(FIXTURE_FILE, "rotation")?,
        table_tolerance: manifest.tolerance(FIXTURE_FILE, "table")?,
    })
}

fn load_case(
    catalog: &SafeTensorCatalog,
    name: &'static str,
    input_name: &str,
    output_name: &str,
    rows: usize,
    head_dim: usize,
) -> Result<RopeCase, BackendError> {
    let input = catalog.load(input_name)?;
    let shape = input.descriptor.layout.shape.clone();
    let &[heads, input_rows, input_head_dim] = shape.as_slice() else {
        return Err(fixture_error(format!(
            "{input_name} must have three dimensions"
        )));
    };
    if heads == 0 || input_rows != rows || input_head_dim != head_dim {
        return Err(fixture_error(format!(
            "{input_name} has shape {shape:?}, expected [heads, {rows}, {head_dim}]"
        )));
    }

    Ok(RopeCase {
        name,
        heads,
        input: input.to_f32_vec()?,
        expected: load_tensor(catalog, output_name, &shape)?,
    })
}

fn load_tensor(
    catalog: &SafeTensorCatalog,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, BackendError> {
    let tensor = catalog.load(name)?;
    if tensor.descriptor.layout.shape != shape {
        return Err(fixture_error(format!(
            "{name} has shape {:?}, expected {shape:?}",
            tensor.descriptor.layout.shape
        )));
    }
    Ok(tensor.to_f32_vec()?)
}

fn fixture_error(message: impl Into<String>) -> BackendError {
    BackendError::RopeFixture(message.into())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
    }

    #[test]
    fn loads_pytorch_rope_cases() {
        let fixtures = load_fixtures(&fixture_directory()).unwrap();

        assert_eq!(fixtures.positions, vec![0, 1, 128, 4096, 32767]);
        assert_eq!(fixtures.head_dim, 128);
        assert_eq!(fixtures.theta, 1.0e6);
        assert_eq!(fixtures.cases[0].heads, 16);
        assert_eq!(fixtures.cases[1].heads, 8);
    }

    #[test]
    fn tables_start_unrotated_at_position_zero() {
        let (cos, sin) = rope_tables(&[0], 8, 1.0e6);

        assert_eq!(cos, vec![1.0; 4]);
        assert_eq!(sin, vec![0.0; 4]);
    }

    #[test]
    fn first_column_turns_one_radian_per_position() {
        let (cos, sin) = rope_tables(&[0, 1, 2], 8, 1.0e6);

        // row 2, column 0
        assert_eq!(cos[8], 2.0_f32.cos());
        assert_eq!(sin[8], 2.0_f32.sin());
    }

    #[test]
    fn tables_match_the_pytorch_fixture() {
        let fixtures = load_fixtures(&fixture_directory()).unwrap();

        check_tables(&fixtures).unwrap();
    }
}
