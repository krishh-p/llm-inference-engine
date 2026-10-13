//! cuTile kernels. They are compiled only with the `cuda` feature, which needs
//! the CUDA toolkit.

#[cfg(feature = "cuda")]
pub use kernels::{
    rms_norm_kernel::{rms_norm_bf16, rms_norm_f32},
    rope_kernel::rope_f32,
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

        /// Rotates one token of one head. The tile holds the two halves of
        /// its vector as two rows, and row `i` of the first half pairs with
        /// row `i` of the second.
        #[cutile::entry()]
        pub fn rope_f32<const HALF: i32>(
            output: &mut Tensor<f32, { [1, 2, HALF] }>,
            input: &Tensor<f32, { [-1, -1, -1] }>,
            cos: &Tensor<f32, { [-1, -1] }>,
            sin: &Tensor<f32, { [-1, -1] }>,
        ) {
            let x: Tile<f32, { [1, 2, HALF] }> = load_tile_like(input, output);
            let first: Tile<i32, { [] }> = scalar_to_tile(0i32);
            let second: Tile<i32, { [] }> = scalar_to_tile(1i32);
            let a: Tile<f32, { [1, 1, HALF] }> = extract(x, [first, first, first]);
            let b: Tile<f32, { [1, 1, HALF] }> = extract(x, [first, second, first]);

            // The tables have one row per token, shared by every head.
            let block: (i32, i32, i32) = get_tile_block_id();
            let cos: Tile<f32, { [1, HALF] }> = load_tile(cos, shape![1, HALF], [block.1, 0i32]);
            let sin: Tile<f32, { [1, HALF] }> = load_tile(sin, shape![1, HALF], [block.1, 0i32]);
            let cos = cos.reshape(shape![1, 1, HALF]);
            let sin = sin.reshape(shape![1, 1, HALF]);

            let rotated: Tile<f32, { [1, 2, HALF] }> =
                cat(a * cos - b * sin, b * cos + a * sin, 1i32);
            output.store(rotated);
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
