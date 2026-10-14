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

/// The shapes of one attention matmul, read from its three tensors.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct AttentionShapes {
    heads: i32,
    kv_heads: i32,
    rows: i32,
    head_dim: i32,
    capacity: i32,
    keys: i32,
}

/// Plans the attention scores of every query head:
/// `scores[h, :, :keys] = q[h] @ k[h / group, :keys]^T`.
///
/// `q` is `[heads, rows, head_dim]`, `k` is a key cache
/// `[kv_heads, capacity, head_dim]`, and `scores` is `[heads, rows, capacity]`.
/// Only the first `keys` cached keys are read, and only the first `keys`
/// columns of each row of scores are written.
///
/// # Errors
///
/// Returns an error when the shapes do not agree or `keys` is zero or more
/// than the cache holds.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn plan_attention_scores(
    q: &[i32],
    k: &[i32],
    scores: &[i32],
    keys: usize,
) -> Result<HeadGemm, BackendError> {
    let shapes = attention_shapes(q, k, scores, true, keys)?;
    let AttentionShapes {
        rows,
        head_dim,
        capacity,
        keys,
        ..
    } = shapes;

    // cuBLAS's A is the keys, read transposed, and its B is the queries.
    // A row of scores is `capacity` wide, of which the first `keys` are written.
    let call = GemmCall {
        transpose_a: true,
        m: keys,
        n: rows,
        k: head_dim,
        lda: head_dim,
        ldb: head_dim,
        ldc: capacity,
    };
    Ok(head_gemm(
        &shapes,
        call,
        [rows * head_dim, capacity * head_dim, rows * capacity],
    ))
}

/// Plans the attention output of every query head:
/// `output[h] = probs[h, :, :keys] @ v[h / group, :keys]`.
///
/// `probs` is `[heads, rows, capacity]`, `v` is a value cache
/// `[kv_heads, capacity, head_dim]`, and `output` is `[heads, rows, head_dim]`.
/// Only the first `keys` columns of probabilities and cached values are read.
///
/// # Errors
///
/// Returns an error when the shapes do not agree or `keys` is zero or more
/// than the cache holds.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn plan_attention_mix(
    probs: &[i32],
    v: &[i32],
    output: &[i32],
    keys: usize,
) -> Result<HeadGemm, BackendError> {
    let shapes = attention_shapes(output, v, probs, false, keys)?;
    let AttentionShapes {
        rows,
        head_dim,
        capacity,
        keys,
        ..
    } = shapes;

    // cuBLAS's A is the values and its B is the probabilities, whose rows
    // are `capacity` wide although only the first `keys` are read.
    let call = GemmCall {
        transpose_a: false,
        m: head_dim,
        n: rows,
        k: keys,
        lda: head_dim,
        ldb: capacity,
        ldc: head_dim,
    };
    Ok(head_gemm(
        &shapes,
        call,
        [rows * capacity, capacity * head_dim, rows * head_dim],
    ))
}

/// Checks the three tensors of an attention matmul against each other.
///
/// `per_head` is `[heads, rows, head_dim]`, `cache` is
/// `[kv_heads, capacity, head_dim]`, and `wide` is `[heads, rows, capacity]`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn attention_shapes(
    per_head: &[i32],
    cache: &[i32],
    wide: &[i32],
    right_transposed: bool,
    keys: usize,
) -> Result<AttentionShapes, BackendError> {
    let mismatch = || BackendError::MatmulShape {
        left: per_head.to_vec(),
        right: cache.to_vec(),
        output: wide.to_vec(),
        right_transposed,
    };

    let (
        &[heads, rows, head_dim],
        &[kv_heads, capacity, cache_dim],
        &[wide_heads, wide_rows, width],
    ) = (per_head, cache, wide)
    else {
        return Err(mismatch());
    };
    let keys = i32::try_from(keys).map_err(|_| mismatch())?;
    let agree = [heads, rows, head_dim, kv_heads, capacity]
        .iter()
        .all(|&dim| dim > 0)
        && heads % kv_heads == 0
        && cache_dim == head_dim
        && [wide_heads, wide_rows, width] == [heads, rows, capacity]
        && keys > 0
        && keys <= capacity;
    if !agree {
        return Err(mismatch());
    }

    Ok(AttentionShapes {
        heads,
        kv_heads,
        rows,
        head_dim,
        capacity,
        keys,
    })
}

/// Turns an attention plan into one GEMM per head. `elements` is how far
/// apart the heads of the left, right and output tensors are.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn head_gemm(shapes: &AttentionShapes, call: GemmCall, elements: [i32; 3]) -> HeadGemm {
    let to_usize = |value: i32| usize::try_from(value).expect("checked to be positive");
    HeadGemm {
        call,
        heads: to_usize(shapes.heads),
        group: to_usize(shapes.heads / shapes.kv_heads),
        left_elements: to_usize(elements[0]),
        right_elements: to_usize(elements[1]),
        output_elements: to_usize(elements[2]),
    }
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
    fn plans_scores_like_a_plain_product_when_the_cache_is_full() {
        // q [16, 5, 128] @ k [8, 5, 128]^T -> [16, 5, 5]
        let plan = plan_attention_scores(&[16, 5, 128], &[8, 5, 128], &[16, 5, 5], 5).unwrap();

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
    fn plans_scores_for_the_filled_part_of_a_cache() {
        // one query against the first 6 of 12 cached keys
        let plan = plan_attention_scores(&[16, 1, 128], &[8, 12, 128], &[16, 1, 12], 6).unwrap();

        assert_eq!(plan.call.m, 6);
        assert_eq!(plan.call.n, 1);
        assert_eq!(plan.call.k, 128);
        // a row of scores is the whole capacity wide
        assert_eq!(plan.call.ldc, 12);
        assert_eq!(plan.right_elements, 12 * 128);
        assert_eq!(plan.output_elements, 12);
    }

    #[test]
    fn plans_the_mix_like_a_plain_product_when_the_cache_is_full() {
        // probs [16, 5, 5] @ v [8, 5, 128] -> [16, 5, 128]
        let plan = plan_attention_mix(&[16, 5, 5], &[8, 5, 128], &[16, 5, 128], 5).unwrap();

        assert_eq!(
            plan.call,
            plan_gemm(&[5, 5], &[5, 128], &[5, 128], false).unwrap()
        );
        assert_eq!(plan.left_elements, 25);
        assert_eq!(plan.right_elements, 640);
        assert_eq!(plan.output_elements, 640);
    }

    #[test]
    fn plans_the_mix_for_the_filled_part_of_a_cache() {
        let plan = plan_attention_mix(&[16, 1, 12], &[8, 12, 128], &[16, 1, 128], 6).unwrap();

        assert_eq!(plan.call.k, 6);
        // a row of probabilities is the whole capacity wide
        assert_eq!(plan.call.ldb, 12);
        assert_eq!(plan.left_elements, 12);
        assert_eq!(plan.right_elements, 12 * 128);
    }

    #[test]
    fn rejects_attention_shapes_that_do_not_agree() {
        // more keys than the cache has, or none
        assert!(plan_attention_scores(&[16, 1, 128], &[8, 12, 128], &[16, 1, 12], 13).is_err());
        assert!(plan_attention_scores(&[16, 1, 128], &[8, 12, 128], &[16, 1, 12], 0).is_err());
        // scores must be as wide as the cache
        assert!(plan_attention_scores(&[16, 1, 128], &[8, 12, 128], &[16, 1, 6], 6).is_err());
        // query heads must be a multiple of the cached heads
        assert!(plan_attention_scores(&[16, 1, 128], &[6, 12, 128], &[16, 1, 12], 6).is_err());
        // the head sizes must match
        assert!(plan_attention_mix(&[16, 1, 12], &[8, 12, 64], &[16, 1, 128], 6).is_err());
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
