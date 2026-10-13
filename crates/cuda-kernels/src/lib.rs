//! cuTile kernels. They are compiled only on Linux, where the CUDA toolkit exists.

#[cfg(target_os = "linux")]
pub use kernels::vector_add_kernel::vector_add;

#[cfg(target_os = "linux")]
mod kernels {
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
