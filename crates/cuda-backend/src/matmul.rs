use crate::BackendError;

/// Arguments for one cuBLAS GEMM call, in cuBLAS's own column-major terms.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GemmCall {
    pub transpose_a: bool,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
}

/// Plans `output = left @ right`, or `left @ right^T`, for row-major matrices.
///
/// cuBLAS reads memory column by column, so it sees a row-major matrix as its
/// transpose. Asking it for `output^T = right^T @ left^T` therefore writes the
/// row-major result directly. The operands swap places: cuBLAS's `A` is `right`
/// and its `B` is `left`, and only `A` is ever transposed.
///
/// # Errors
///
/// Returns an error when a shape is not a non-empty matrix or the three shapes
/// do not agree.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn plan_gemm(
    left: &[i32],
    right: &[i32],
    output: &[i32],
    right_transposed: bool,
) -> Result<GemmCall, BackendError> {
    let mismatch = || BackendError::MatmulShape {
        left: left.to_vec(),
        right: right.to_vec(),
        output: output.to_vec(),
        right_transposed,
    };

    let (&[rows, inner], &[right_rows, right_cols], &[output_rows, cols]) = (left, right, output)
    else {
        return Err(mismatch());
    };
    if [rows, inner, right_rows, right_cols, output_rows, cols]
        .iter()
        .any(|&dimension| dimension <= 0)
    {
        return Err(mismatch());
    }

    // The mathematical right operand is [inner, cols]. When it is stored
    // transposed, its buffer is [cols, inner].
    let expected_right = if right_transposed {
        [cols, inner]
    } else {
        [inner, cols]
    };
    if [right_rows, right_cols] != expected_right || output_rows != rows {
        return Err(mismatch());
    }

    Ok(GemmCall {
        transpose_a: right_transposed,
        m: cols,
        n: rows,
        k: inner,
        lda: right_cols,
        ldb: inner,
        ldc: cols,
    })
}

/// Checks that a matrix lives on the device the backend runs on.
///
/// # Errors
///
/// Returns an error naming the matrix when the two devices differ.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_device(
    name: &'static str,
    matrix_device: usize,
    backend_device: usize,
) -> Result<(), BackendError> {
    if matrix_device != backend_device {
        return Err(BackendError::WrongDevice {
            name,
            expected: backend_device,
            actual: matrix_device,
        });
    }
    Ok(())
}

/// Multiplies row-major matrices on the host: `left @ right`, or `left @ right^T`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn reference(
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    right_transposed: bool,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0_f32;
            for index in 0..inner {
                let right_value = if right_transposed {
                    right[col * inner + index]
                } else {
                    right[index * cols + col]
                };
                sum += left[row * inner + index] * right_value;
            }
            output.push(sum);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_a_plain_product() {
        // [2, 3] @ [3, 4] -> [2, 4]
        let call = plan_gemm(&[2, 3], &[3, 4], &[2, 4], false).unwrap();

        assert_eq!(
            call,
            GemmCall {
                transpose_a: false,
                m: 4,
                n: 2,
                k: 3,
                lda: 4,
                ldb: 3,
                ldc: 4,
            }
        );
    }

    #[test]
    fn plans_a_transposed_right_operand() {
        // [2, 3] @ [4, 3]^T -> [2, 4]
        let call = plan_gemm(&[2, 3], &[4, 3], &[2, 4], true).unwrap();

        assert_eq!(
            call,
            GemmCall {
                transpose_a: true,
                m: 4,
                n: 2,
                k: 3,
                lda: 3,
                ldb: 3,
                ldc: 4,
            }
        );
    }

    #[test]
    fn rejects_shapes_that_only_fit_the_other_layout() {
        // [4, 3] is the transposed layout, so the plain product must refuse it.
        assert!(matches!(
            plan_gemm(&[2, 3], &[4, 3], &[2, 4], false),
            Err(BackendError::MatmulShape { .. })
        ));
        assert!(matches!(
            plan_gemm(&[2, 3], &[3, 4], &[2, 4], true),
            Err(BackendError::MatmulShape { .. })
        ));
    }

    #[test]
    fn rejects_a_wrong_output_shape() {
        assert!(matches!(
            plan_gemm(&[2, 3], &[3, 4], &[4, 2], false),
            Err(BackendError::MatmulShape { .. })
        ));
    }

    #[test]
    fn rejects_shapes_that_are_not_matrices() {
        assert!(matches!(
            plan_gemm(&[6], &[3, 4], &[2, 4], false),
            Err(BackendError::MatmulShape { .. })
        ));
        assert!(matches!(
            plan_gemm(&[2, 0], &[0, 4], &[2, 4], false),
            Err(BackendError::MatmulShape { .. })
        ));
    }

    #[test]
    fn accepts_a_matrix_on_the_backend_device() {
        assert!(check_device("left", 0, 0).is_ok());
    }

    #[test]
    fn rejects_a_matrix_on_another_device() {
        let error = check_device("right", 1, 0).unwrap_err();

        assert_eq!(
            error.to_string(),
            "matmul right matrix is on CUDA device 1, but the backend uses device 0"
        );
    }

    #[test]
    fn reference_multiplies_non_square_matrices() {
        // [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
        let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        assert_eq!(
            reference(&left, &right, 2, 3, 2, false),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn reference_reads_a_transposed_right_operand() {
        // the same right matrix as above, stored as its transpose
        let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right_transposed = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];

        assert_eq!(
            reference(&left, &right_transposed, 2, 3, 2, true),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }
}
