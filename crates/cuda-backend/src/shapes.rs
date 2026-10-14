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

/// Checks that `rows` rows starting at `first` fit in a matrix of `available` rows.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_row_range(
    operation: &'static str,
    name: &'static str,
    first: usize,
    rows: usize,
    available: usize,
) -> Result<(), BackendError> {
    // written so it cannot overflow, since this guards a raw memory copy
    if rows == 0 || first > available || rows > available - first {
        return Err(BackendError::RowRange {
            operation,
            name,
            first,
            rows,
            available,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_last_row() {
        assert!(check_row_range("copy_rows", "input", 4, 1, 5).is_ok());
    }

    #[test]
    fn rejects_rows_past_the_end() {
        let error = check_row_range("copy_rows", "input", 4, 2, 5).unwrap_err();

        assert_eq!(
            error.to_string(),
            "copy_rows input needs 2 rows from row 4, but has 5 rows"
        );
        assert!(check_row_range("copy_rows", "input", 0, 0, 5).is_err());
    }

    #[test]
    fn rejects_ranges_that_would_overflow() {
        assert!(check_row_range("copy_rows", "input", usize::MAX, 1, 5).is_err());
        assert!(check_row_range("copy_rows", "input", 1, usize::MAX, 5).is_err());
        assert!(check_row_range("copy_rows", "input", usize::MAX, usize::MAX, usize::MAX).is_err());
    }

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
