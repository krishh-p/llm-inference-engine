use std::sync::Arc;

use cuda_core::{Device, Stream};
use cuda_kernels::{causal_softmax_f32, rms_norm_f32, rope_f32, swap_leading_axes_f32};
use cutile::prelude::*;

use crate::cublas::CublasHandle;
use crate::matmul::{check_device, plan_gemm, plan_head_gemm};
use crate::rms_norm::inverse_hidden_size;
use crate::shapes::{dims, expect_shape};
use crate::{BackendError, check_rms_norm_shape};

/// One CUDA device with the stream and cuBLAS handle that every operation uses.
///
/// Each method waits for its work to finish before it returns, so a result can
/// be read, and an input dropped, as soon as the call is over.
pub struct Backend {
    pub(crate) device: Arc<Device>,
    pub(crate) stream: Arc<Stream>,
    cublas: CublasHandle,
}

impl Backend {
    /// Opens the first CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error when the device, its stream or cuBLAS cannot be opened.
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::new(0).map_err(cuda_error)?;
        let stream = device.new_stream().map_err(cuda_error)?;
        let cublas = CublasHandle::new(&stream)?;

        Ok(Self {
            device,
            stream,
            cublas,
        })
    }

    /// Copies row-major float32 values to the device.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` does not fill `shape` or the copy fails.
    pub fn upload(&self, values: &[f32], shape: &[usize]) -> Result<Tensor<f32>, BackendError> {
        api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
            .sync_on(&self.stream)
            .map_err(cutile_error)?
            .reshape(shape)
            .map_err(cutile_error)
    }

    /// Allocates a float32 tensor of zeros on the device.
    ///
    /// # Errors
    ///
    /// Returns an error when the allocation fails.
    pub fn zeros(&self, shape: &[usize]) -> Result<Tensor<f32>, BackendError> {
        api::zeros::<f32>(shape)
            .sync_on(&self.stream)
            .map_err(cutile_error)
    }

    /// Copies a tensor back to host memory in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error when the copy fails.
    pub fn download(&self, tensor: &Tensor<f32>) -> Result<Vec<f32>, BackendError> {
        // cuTile's host copy takes ownership, so it is given a device copy.
        let copy = tensor.dup().sync_on(&self.stream).map_err(cutile_error)?;
        copy.to_host_vec()
            .sync_on(&self.stream)
            .map_err(cutile_error)
    }

    /// Computes `output = left @ right` for row-major matrices.
    ///
    /// # Errors
    ///
    /// Returns an error when a matrix is on another device or is not
    /// contiguous, the shapes do not agree, or cuBLAS fails.
    pub fn matmul(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        self.gemm(output, left, right, false)
    }

    /// Computes `output = left @ right^T` for row-major matrices.
    ///
    /// # Errors
    ///
    /// Returns an error when a matrix is on another device or is not
    /// contiguous, the shapes do not agree, or cuBLAS fails.
    pub fn matmul_transposed(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        self.gemm(output, left, right, true)
    }

    /// Computes `output[h] = left[h] @ right[h / group]` for every head `h`.
    ///
    /// The tensors are `[heads, rows, cols]`. When `right` has fewer heads,
    /// each one is shared by `group` consecutive left heads, which is how
    /// grouped-query attention pairs query heads with key and value heads.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device or is not
    /// contiguous, the shapes do not agree, or cuBLAS fails.
    pub fn matmul_heads(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        self.head_gemm(output, left, right, false, 1.0)
    }

    /// Computes `output[h] = scale * left[h] @ right[h / group]^T` for every
    /// head `h`. See [`Backend::matmul_heads`] for how heads are paired.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device or is not
    /// contiguous, the shapes do not agree, or cuBLAS fails.
    pub fn matmul_heads_transposed(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
        scale: f32,
    ) -> Result<(), BackendError> {
        self.head_gemm(output, left, right, true, scale)
    }

    /// Normalizes each row of a `[rows, hidden_size]` matrix and scales it by `weight`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, a shape is wrong
    /// or unsupported, or the kernel fails.
    pub fn rms_norm(
        &self,
        output: &mut Tensor<f32>,
        input: &Tensor<f32>,
        weight: &Tensor<f32>,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        let [rows, hidden_size] = dims("rms_norm", "input", input.shape())?;
        check_rms_norm_shape(rows, hidden_size)?;
        expect_shape("rms_norm", "output", output.shape(), &[rows, hidden_size])?;
        expect_shape("rms_norm", "weight", weight.shape(), &[hidden_size])?;
        self.check_tensors(&[("input", input), ("weight", weight), ("output", &*output)])?;

        // The tile width is rounded up to a power of two. The padding adds
        // nothing to the sum, and the mean still divides by the real width.
        let _ = rms_norm_f32(
            output.partition([1, hidden_size.next_power_of_two()]),
            input,
            weight,
            inverse_hidden_size(hidden_size),
            epsilon,
        )
        .sync_on(&self.stream)
        .map_err(cutile_error)?;
        Ok(())
    }

    /// Rotates `[heads, rows, head_dim]` vectors by their token positions.
    ///
    /// `cos` and `sin` are `[rows, head_dim / 2]` tables with one row for the
    /// position of each token.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree, or the kernel fails.
    pub fn rope(
        &self,
        output: &mut Tensor<f32>,
        input: &Tensor<f32>,
        cos: &Tensor<f32>,
        sin: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [heads, rows, head_dim] = dims("rope", "input", input.shape())?;
        let half = head_dim / 2;
        expect_shape("rope", "input", input.shape(), &[heads, rows, half * 2])?;
        expect_shape("rope", "output", output.shape(), &[heads, rows, head_dim])?;
        expect_shape("rope", "cos", cos.shape(), &[rows, half])?;
        expect_shape("rope", "sin", sin.shape(), &[rows, half])?;
        self.check_tensors(&[
            ("input", input),
            ("cos", cos),
            ("sin", sin),
            ("output", &*output),
        ])?;

        let _ = rope_f32(output.partition([1, 1, half]), input, cos, sin)
            .sync_on(&self.stream)
            .map_err(cutile_error)?;
        Ok(())
    }

    /// Copies `input[b, a, :]` to `output[a, b, :]`.
    ///
    /// This turns `[rows, heads, head_dim]` into `[heads, rows, head_dim]` and back.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree, or the kernel fails.
    pub fn swap_leading_axes(
        &self,
        output: &mut Tensor<f32>,
        input: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [first, second, width] = dims("swap_leading_axes", "input", input.shape())?;
        expect_shape(
            "swap_leading_axes",
            "output",
            output.shape(),
            &[second, first, width],
        )?;
        self.check_tensors(&[("input", input), ("output", &*output)])?;

        let _ = swap_leading_axes_f32(output.partition([1, 1, width]), input)
            .sync_on(&self.stream)
            .map_err(cutile_error)?;
        Ok(())
    }

    /// Turns `[heads, rows, cols]` attention scores into probabilities.
    ///
    /// Row `r` belongs to the token at `first_position + r` and column `c` to
    /// the token at position `c`. A token sees itself and earlier tokens only.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree with `first_position`, or the kernel fails.
    pub fn causal_softmax(
        &self,
        output: &mut Tensor<f32>,
        scores: &Tensor<f32>,
        first_position: usize,
    ) -> Result<(), BackendError> {
        let [heads, rows, cols] = dims("causal_softmax", "scores", scores.shape())?;
        // the last row is the newest token, so it must see every column
        expect_shape(
            "causal_softmax",
            "scores",
            scores.shape(),
            &[heads, rows, first_position + rows],
        )?;
        expect_shape(
            "causal_softmax",
            "output",
            output.shape(),
            &[heads, rows, cols],
        )?;
        self.check_tensors(&[("scores", scores), ("output", &*output)])?;
        // the shape check above showed this difference is `first_position`
        let first_position = scores.shape()[2] - scores.shape()[1];

        // Columns past the real width are padding. They sit after the newest
        // token, so the causal mask hides them too.
        let _ = causal_softmax_f32(
            output.partition([1, 1, cols.next_power_of_two()]),
            scores,
            first_position,
        )
        .sync_on(&self.stream)
        .map_err(cutile_error)?;
        Ok(())
    }

    #[allow(unsafe_code)]
    fn gemm(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
        right_transposed: bool,
    ) -> Result<(), BackendError> {
        self.check_tensors(&[("left", left), ("right", right), ("output", &*output)])?;
        let call = plan_gemm(
            left.shape(),
            right.shape(),
            output.shape(),
            right_transposed,
        )?;

        // SAFETY: the three tensors were checked to be contiguous float32
        // buffers on this device, plan_gemm checked that their shapes agree,
        // and they stay borrowed until the stream has been drained below.
        unsafe {
            // cuBLAS's first operand is the right matrix. See plan_gemm.
            self.cublas.gemm_f32(
                &call,
                1.0,
                device_address(right),
                device_address(left),
                device_address(output),
            )?;
        }
        self.wait_for_stream()
    }

    #[allow(unsafe_code)]
    fn head_gemm(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
        right_transposed: bool,
        scale: f32,
    ) -> Result<(), BackendError> {
        self.check_tensors(&[("left", left), ("right", right), ("output", &*output)])?;
        let plan = plan_head_gemm(
            left.shape(),
            right.shape(),
            output.shape(),
            right_transposed,
        )?;

        for head in 0..plan.heads {
            let left_offset = head * plan.left_elements * size_of::<f32>();
            let right_offset = (head / plan.group) * plan.right_elements * size_of::<f32>();
            let output_offset = head * plan.output_elements * size_of::<f32>();

            // SAFETY: the three tensors were checked to be contiguous float32
            // buffers on this device, plan_head_gemm checked that every head
            // lies inside its tensor, and the tensors stay borrowed until the
            // stream has been drained below.
            unsafe {
                self.cublas.gemm_f32(
                    &plan.call,
                    scale,
                    device_address(right) + right_offset,
                    device_address(left) + left_offset,
                    device_address(output) + output_offset,
                )?;
            }
        }
        self.wait_for_stream()
    }

    /// Checks that every tensor is contiguous and on this backend's device.
    /// A device address only means something on the device that allocated it.
    fn check_tensors(&self, tensors: &[(&'static str, &Tensor<f32>)]) -> Result<(), BackendError> {
        for &(name, tensor) in tensors {
            check_device(name, tensor.device_id(), self.device.ordinal())?;
            if !tensor.is_contiguous() {
                return Err(BackendError::NonContiguous(name));
            }
        }
        Ok(())
    }

    /// Blocks until the stream has finished everything queued on it.
    fn wait_for_stream(&self) -> Result<(), BackendError> {
        let finished = self.device.new_event().map_err(cuda_error)?;
        finished.record(&self.stream).map_err(cuda_error)?;
        finished.synchronize().map_err(cuda_error)
    }
}

/// Gives a tensor a new shape with the same number of elements.
///
/// # Errors
///
/// Returns an error when `shape` holds a different number of elements.
pub fn reshape(tensor: Tensor<f32>, shape: &[usize]) -> Result<Tensor<f32>, BackendError> {
    tensor.reshape(shape).map_err(cutile_error)
}

fn device_address(tensor: &Tensor<f32>) -> usize {
    usize::try_from(tensor.device_pointer().cu_deviceptr())
        .expect("CUDA device addresses fit in usize")
}

fn cuda_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::Cuda(error.to_string())
}

fn cutile_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::Cutile(error.to_string())
}
