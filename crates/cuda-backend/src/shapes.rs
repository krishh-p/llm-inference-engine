use crate::BackendError;

/// Reads a tensor's dimensions after checking how many it has.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn dims<const N: usize>(
    operation: &'static str,
    name: &'static str,
    shape: &[i32],
) -> Result<[usize; N], BackendError> {
    let dims: [i32; N] = shape.try_into().map_err(|_| BackendError::Rank {
        operation,
        name,
        expected: N,
        actual: shape.to_vec(),
    })?;
    Ok(dims.map(|dim| usize::try_from(dim).expect("tensor dimensions are never negative")))
}

/// Checks that a tensor has exactly the expected shape.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn expect_shape(
    operation: &'static str,
    name: &'static str,
    actual: &[i32],
    expected: &[usize],
) -> Result<(), BackendError> {
    let matches = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&actual, &expected)| usize::try_from(actual) == Ok(expected));
    if !matches {
        return Err(BackendError::Shape {
            operation,
            name,
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_dimensions_of_the_expected_rank() {
        assert_eq!(
            dims::<3>("rope", "input", &[16, 5, 128]).unwrap(),
            [16, 5, 128]
        );
    }

    #[test]
    fn rejects_another_rank() {
        let error = dims::<3>("rope", "input", &[5, 2048]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "rope input has shape [5, 2048], expected 3 dimensions"
        );
    }

    #[test]
    fn accepts_the_expected_shape() {
        assert!(expect_shape("rope", "cos", &[5, 64], &[5, 64]).is_ok());
    }

    #[test]
    fn rejects_another_shape() {
        let error = expect_shape("rope", "cos", &[5, 128], &[5, 64]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "rope cos has shape [5, 128], expected [5, 64]"
        );
        assert!(expect_shape("rope", "cos", &[5], &[5, 64]).is_err());
    }
}
