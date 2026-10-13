use std::sync::Arc;

use cuda_core::{Device, Stream};
use cutile::prelude::*;

use crate::BackendError;
use crate::cublas::CublasHandle;
use crate::matmul::{check_device, plan_gemm};

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

    /// Copies a row-major float32 matrix to the device.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` does not fill `shape` or the copy fails.
    pub fn upload_matrix(
        &self,
        values: &[f32],
        shape: [usize; 2],
    ) -> Result<Tensor<f32>, BackendError> {
        api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
            .sync_on(&self.stream)
            .map_err(cutile_error)?
            .reshape(&shape)
            .map_err(cutile_error)
    }

    /// Allocates a float32 matrix of zeros on the device.
    ///
    /// # Errors
    ///
    /// Returns an error when the allocation fails.
    pub fn zeros_matrix(&self, shape: [usize; 2]) -> Result<Tensor<f32>, BackendError> {
        api::zeros::<f32>(&shape)
            .sync_on(&self.stream)
            .map_err(cutile_error)
    }

    /// Copies a tensor back to host memory in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error when the copy fails.
    pub fn download(&self, tensor: Tensor<f32>) -> Result<Vec<f32>, BackendError> {
        tensor
            .to_host_vec()
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

    #[allow(unsafe_code)]
    fn gemm(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
        right_transposed: bool,
    ) -> Result<(), BackendError> {
        for (name, tensor) in [("left", left), ("right", right), ("output", &*output)] {
            // A device address only means something in the context that
            // allocated it, so a matrix from another device must be refused.
            check_device(name, tensor.device_id(), self.device.ordinal())?;
            if !tensor.is_contiguous() {
                return Err(BackendError::NonContiguousMatrix(name));
            }
        }
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
                device_address(right),
                device_address(left),
                device_address(output),
            )?;
        }
        self.wait_for_stream()
    }

    /// Blocks until the stream has finished everything queued on it.
    fn wait_for_stream(&self) -> Result<(), BackendError> {
        let finished = self.device.new_event().map_err(cuda_error)?;
        finished.record(&self.stream).map_err(cuda_error)?;
        finished.synchronize().map_err(cuda_error)
    }
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
