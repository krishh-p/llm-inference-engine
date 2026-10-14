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

/// One GEMM per head of a `[heads, rows, cols]` tensor.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HeadGemm {
    pub call: GemmCall,
    pub heads: usize,
    /// How many left heads share each right head.
    pub group: usize,
    /// Elements from the start of one head to the start of the next.
    pub left_elements: usize,
    pub right_elements: usize,
    pub output_elements: usize,
}

/// Plans `output[h] = left[h] @ right[h / group]` for every head `h`, with the
/// right matrices optionally transposed.
///
/// Only the first `right_rows` rows of each right head are used, and the rest
/// are skipped. A KV cache keeps room for more tokens than it holds, so this
/// lets attention read just the filled rows.
///
/// # Errors
///
/// Returns an error when a shape is not three-dimensional, the left heads are
/// not a multiple of the right heads, `right_rows` is zero or more than the
/// right heads have, or the matrices in each head do not agree.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn plan_head_gemm(
    left: &[i32],
    right: &[i32],
    output: &[i32],
    right_transposed: bool,
    right_rows: usize,
) -> Result<HeadGemm, BackendError> {
    let mismatch = || BackendError::MatmulShape {
        left: left.to_vec(),
        right: right.to_vec(),
        output: output.to_vec(),
        right_transposed,
    };

    let (&[heads, ..], &[right_heads, ..], &[output_heads, ..]) = (left, right, output) else {
        return Err(mismatch());
    };
    if left.len() != 3 || right.len() != 3 || output.len() != 3 {
        return Err(mismatch());
    }
    if right_heads <= 0 || heads % right_heads != 0 || output_heads != heads {
        return Err(mismatch());
    }
    let used_rows = i32::try_from(right_rows).map_err(|_| mismatch())?;
    if used_rows <= 0 || used_rows > right[1] {
        return Err(mismatch());
    }
    let used_right = [used_rows, right[2]];
    let call = plan_gemm(&left[1..], &used_right, &output[1..], right_transposed)
        .map_err(|_| mismatch())?;

    let elements = |shape: &[i32]| {
        usize::try_from(shape[1] * shape[2]).expect("plan_gemm checked the dimensions are positive")
    };
    Ok(HeadGemm {
        call,
        heads: usize::try_from(heads).expect("checked to be positive"),
        group: usize::try_from(heads / right_heads).expect("checked to be positive"),
        left_elements: elements(left),
        right_elements: elements(right),
        output_elements: elements(output),
    })
}

/// Checks that a tensor lives on the device the backend runs on.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_device(
    name: &'static str,
    tensor_device: usize,
    backend_device: usize,
) -> Result<(), BackendError> {
    if tensor_device != backend_device {
        return Err(BackendError::WrongDevice {
            name,
            expected: backend_device,
            actual: tensor_device,
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
    fn plans_attention_scores_with_shared_key_heads() {
        // q [16, 5, 128] @ k [8, 5, 128]^T -> [16, 5, 5]
        let plan = plan_head_gemm(&[16, 5, 128], &[8, 5, 128], &[16, 5, 5], true, 5).unwrap();

        assert_eq!(
            plan.call,
            plan_gemm(&[5, 128], &[5, 128], &[5, 5], true).unwrap()
        );
        assert_eq!(plan.heads, 16);
        assert_eq!(plan.group, 2);
        assert_eq!(plan.left_elements, 640);
        assert_eq!(plan.right_elements, 640);
        assert_eq!(plan.output_elements, 25);
    }

    #[test]
    fn rejects_heads_that_do_not_divide_evenly() {
        assert!(matches!(
            plan_head_gemm(&[16, 5, 128], &[6, 5, 128], &[16, 5, 5], true, 5),
            Err(BackendError::MatmulShape { .. })
        ));
        assert!(matches!(
            plan_head_gemm(&[16, 5, 128], &[8, 5, 128], &[8, 5, 5], true, 5),
            Err(BackendError::MatmulShape { .. })
        ));
        assert!(matches!(
            plan_head_gemm(&[5, 128], &[5, 128], &[5, 5], true, 5),
            Err(BackendError::MatmulShape { .. })
        ));
    }

    #[test]
    fn reads_only_the_filled_rows_of_a_cache() {
        // q [16, 1, 128] @ the first 6 of 12 rows of k [8, 12, 128], transposed
        let plan = plan_head_gemm(&[16, 1, 128], &[8, 12, 128], &[16, 1, 6], true, 6).unwrap();

        assert_eq!(
            plan.call,
            plan_gemm(&[1, 128], &[6, 128], &[1, 6], true).unwrap()
        );
        // the next head still starts a whole cache head later
        assert_eq!(plan.right_elements, 12 * 128);
        assert_eq!(plan.output_elements, 6);
    }

    #[test]
    fn mixes_values_from_the_filled_rows_of_a_cache() {
        // probs [16, 1, 6] @ the first 6 of 12 rows of v [8, 12, 128]
        let plan = plan_head_gemm(&[16, 1, 6], &[8, 12, 128], &[16, 1, 128], false, 6).unwrap();

        assert_eq!(
            plan.call,
            plan_gemm(&[1, 6], &[6, 128], &[1, 128], false).unwrap()
        );
        assert_eq!(plan.right_elements, 12 * 128);
    }

    #[test]
    fn rejects_more_rows_than_the_cache_has() {
        assert!(matches!(
            plan_head_gemm(&[16, 1, 128], &[8, 12, 128], &[16, 1, 13], true, 13),
            Err(BackendError::MatmulShape { .. })
        ));
        assert!(matches!(
            plan_head_gemm(&[16, 1, 128], &[8, 12, 128], &[16, 1, 0], true, 0),
            Err(BackendError::MatmulShape { .. })
        ));
        // the output must match the rows that are read, not the whole cache
        assert!(matches!(
            plan_head_gemm(&[16, 1, 128], &[8, 12, 128], &[16, 1, 12], true, 6),
            Err(BackendError::MatmulShape { .. })
        ));
    }

    #[test]
    fn accepts_a_tensor_on_the_backend_device() {
        assert!(check_device("left", 0, 0).is_ok());
    }

    #[test]
    fn rejects_a_tensor_on_another_device() {
        let error = check_device("right", 1, 0).unwrap_err();

        assert_eq!(
            error.to_string(),
            "right tensor is on CUDA device 1, but the backend uses device 0"
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
