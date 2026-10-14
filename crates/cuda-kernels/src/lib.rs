//! cuTile kernels. They are compiled only with the `cuda` feature, which needs
//! the CUDA toolkit.

#[cfg(feature = "cuda")]
pub use kernels::{
    causal_softmax_kernel::causal_softmax_f32,
    concat_rows_kernel::concat_rows_f32,
    elementwise_kernel::{add_f32, silu_mul_f32},
    rms_norm_kernel::{rms_norm_bf16, rms_norm_f32},
    rope_kernel::rope_f32,
    swap_leading_axes_kernel::swap_leading_axes_f32,
    vector_add_kernel::vector_add,
};

#[cfg(feature = "cuda")]
mod kernels {
    #[cutile::module]
    pub mod rms_norm_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        #[cutile::entry()]
        pub fn rms_norm_f32<const H: i32>(
            output: &mut Tensor<f32, { [1, H] }>,
            input: &Tensor<f32, { [-1, -1] }>,
            weight: &Tensor<f32, { [-1] }>,
            inverse_hidden_size: f32,
            epsilon: f32,
        ) {
            let x: Tile<f32, { [1, H] }> = load_tile_like(input, output);
            let sum: Tile<f32, { [1] }> = reduce_sum(x * x, 1);
            let inverse_hidden_size: Tile<f32, { [1] }> =
                broadcast_scalar(inverse_hidden_size, shape![1]);
            let epsilon: Tile<f32, { [1] }> = broadcast_scalar(epsilon, shape![1]);
            let inverse_rms = rsqrt(sum * inverse_hidden_size + epsilon, ftz::Disabled);
            let inverse_rms = inverse_rms.reshape(shape![1, 1]).broadcast(output.shape());

            let weight: Tile<f32, { [H] }> = load_tile(weight, shape![H], [0]);
            let weight = weight.reshape(shape![1, H]);
            output.store(x * inverse_rms * weight);
        }

        #[cutile::entry()]
        pub fn rms_norm_bf16<const H: i32>(
            output: &mut Tensor<bf16, { [1, H] }>,
            input: &Tensor<bf16, { [-1, -1] }>,
            weight: &Tensor<bf16, { [-1] }>,
            inverse_hidden_size: f32,
            epsilon: f32,
        ) {
            let x: Tile<bf16, { [1, H] }> = load_tile_like(input, output);
            let x_f32: Tile<f32, { [1, H] }> = convert_tile(x);
            let sum: Tile<f32, { [1] }> = reduce_sum(x_f32 * x_f32, 1);
            let inverse_hidden_size: Tile<f32, { [1] }> =
                broadcast_scalar(inverse_hidden_size, shape![1]);
            let epsilon: Tile<f32, { [1] }> = broadcast_scalar(epsilon, shape![1]);
            let inverse_rms = rsqrt(sum * inverse_hidden_size + epsilon, ftz::Disabled);
            let inverse_rms = inverse_rms.reshape(shape![1, 1]).broadcast(output.shape());

            let normalized: Tile<bf16, { [1, H] }> = convert_tile(x_f32 * inverse_rms);
            let weight: Tile<bf16, { [H] }> = load_tile(weight, shape![H], [0]);
            let weight = weight.reshape(shape![1, H]);
            output.store(normalized * weight);
        }
    }

    #[cutile::module]
    pub mod rope_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        /// Rotates one half of one token's vector. Element `i` of the first
        /// half pairs with element `i` of the second, so each tile reads its
        /// own half and the other one.
        #[cutile::entry()]
        pub fn rope_f32<const HALF: i32>(
            output: &mut Tensor<f32, { [1, 1, HALF] }>,
            input: &Tensor<f32, { [-1, -1, -1] }>,
            cos: &Tensor<f32, { [-1, -1] }>,
            sin: &Tensor<f32, { [-1, -1] }>,
        ) {
            let block: (i32, i32, i32) = get_tile_block_id();
            let own: Tile<f32, { [1, 1, HALF] }> =
                load_tile(input, shape![1, 1, HALF], [block.0, block.1, block.2]);
            let other: Tile<f32, { [1, 1, HALF] }> = load_tile(
                input,
                shape![1, 1, HALF],
                [block.0, block.1, 1i32 - block.2],
            );

            // The tables have one row per token, shared by every head.
            let cos: Tile<f32, { [1, HALF] }> = load_tile(cos, shape![1, HALF], [block.1, 0i32]);
            let sin: Tile<f32, { [1, HALF] }> = load_tile(sin, shape![1, HALF], [block.1, 0i32]);
            let cos = cos.reshape(shape![1, 1, HALF]);
            let sin = sin.reshape(shape![1, 1, HALF]);

            // first half: a * cos - b * sin, second half: b * cos + a * sin
            let direction: f32 = convert_scalar::<f32>(2i32 * block.2 - 1i32);
            let direction: Tile<f32, { [1, 1, HALF] }> =
                broadcast_scalar(direction, shape![1, 1, HALF]);
            output.store(own * cos + other * sin * direction);
        }
    }

    #[cutile::module]
    pub mod swap_leading_axes_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        /// Copies `input[b, a, :]` to `output[a, b, :]`.
        #[cutile::entry()]
        pub fn swap_leading_axes_f32<const D: i32>(
            output: &mut Tensor<f32, { [1, 1, D] }>,
            input: &Tensor<f32, { [-1, -1, -1] }>,
        ) {
            let block: (i32, i32, i32) = get_tile_block_id();
            let x: Tile<f32, { [1, 1, D] }> =
                load_tile(input, shape![1, 1, D], [block.1, block.0, 0i32]);
            output.store(x);
        }
    }

    #[cutile::module]
    pub mod causal_softmax_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        /// Softmax over one row of attention scores. Row `r` belongs to the
        /// token at `first_position + r`, which may only see columns up to
        /// its own position.
        #[cutile::entry()]
        pub fn causal_softmax_f32<const N: i32>(
            output: &mut Tensor<f32, { [1, 1, N] }>,
            scores: &Tensor<f32, { [-1, -1, -1] }>,
            first_position: i32,
        ) {
            let block: (i32, i32, i32) = get_tile_block_id();
            let x: Tile<f32, { [1, 1, N] }> = load_tile_like(scores, output);

            let column: Tile<i32, { [N] }> = iota(shape![N]);
            let column = column.reshape(shape![1, 1, N]);
            let last_visible: Tile<i32, { [1, 1, N] }> =
                broadcast_scalar(first_position + block.1, shape![1, 1, N]);
            let visible: Tile<bool, { [1, 1, N] }> = le_tile(column, last_visible);
            let hidden: Tile<f32, { [1, 1, N] }> = constant(f32::NEG_INFINITY, shape![1, 1, N]);
            let x = select(visible, x, hidden);

            let max: Tile<f32, { [1, 1] }> = reduce_max(x, 2);
            let max = max.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, N]);
            let weight = exp(x - max);
            let sum: Tile<f32, { [1, 1] }> = reduce_sum(weight, 2);
            let sum = sum.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, N]);
            output.store(weight / sum);
        }
    }

    #[cutile::module]
    pub mod concat_rows_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        /// Joins two `[heads, rows, width]` tensors along their rows. The
        /// first `first_rows` rows of each head come from `first`.
        #[cutile::entry()]
        pub fn concat_rows_f32<const D: i32>(
            output: &mut Tensor<f32, { [1, 1, D] }>,
            first: &Tensor<f32, { [-1, -1, -1] }>,
            second: &Tensor<f32, { [-1, -1, -1] }>,
            first_rows: i32,
        ) {
            let block: (i32, i32, i32) = get_tile_block_id();
            // rows that belong to `first` read row 0 here, then replace it below
            let second_row: i32 = max(block.1 - first_rows, 0i32);
            let mut x: Tile<f32, { [1, 1, D] }> =
                load_tile(second, shape![1, 1, D], [block.0, second_row, 0i32]);
            if block.1 < first_rows {
                x = load_tile(first, shape![1, 1, D], [block.0, block.1, 0i32]);
            }
            output.store(x);
        }
    }

    #[cutile::module]
    pub mod elementwise_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        /// Adds two matrices, one row per tile.
        #[cutile::entry()]
        pub fn add_f32<const W: i32>(
            output: &mut Tensor<f32, { [1, W] }>,
            left: &Tensor<f32, { [-1, -1] }>,
            right: &Tensor<f32, { [-1, -1] }>,
        ) {
            let left: Tile<f32, { [1, W] }> = load_tile_like(left, output);
            let right: Tile<f32, { [1, W] }> = load_tile_like(right, output);
            output.store(left + right);
        }

        /// Computes `silu(gate) * up`, where `silu(x) = x * sigmoid(x)`.
        #[cutile::entry()]
        pub fn silu_mul_f32<const W: i32>(
            output: &mut Tensor<f32, { [1, W] }>,
            gate: &Tensor<f32, { [-1, -1] }>,
            up: &Tensor<f32, { [-1, -1] }>,
        ) {
            let gate: Tile<f32, { [1, W] }> = load_tile_like(gate, output);
            let up: Tile<f32, { [1, W] }> = load_tile_like(up, output);

            let zero: Tile<f32, { [1, W] }> = constant(0.0f32, shape![1, W]);
            let one: Tile<f32, { [1, W] }> = constant(1.0f32, shape![1, W]);
            let sigmoid = one / (one + exp(zero - gate));
            output.store(gate * sigmoid * up);
        }
    }

    #[cutile::module]
    pub mod vector_add_kernel {
        #[allow(clippy::wildcard_imports)]
        use cutile::core::*;

        #[cutile::entry()]
        pub fn vector_add<const S: [i32; 1]>(
            output: &mut Tensor<f32, S>,
            left: &Tensor<f32, { [-1] }>,
            right: &Tensor<f32, { [-1] }>,
        ) {
            let left_tile = load_tile_like(left, output);
            let right_tile = load_tile_like(right, output);
            output.store(left_tile + right_tile);
        }
    }
}
