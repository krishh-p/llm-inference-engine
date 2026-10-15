//! Private bindings for the cuBLAS calls the backend uses.
//!
//! The pinned CUDA crates do not cover cuBLAS, so its functions are declared
//! here and wrapped. This is the only module that calls them.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_void};
use std::sync::Arc;

use cuda_core::Stream;

use crate::BackendError;
use crate::matmul::GemmCall;

// values from cublas_api.h and library_types.h in CUDA 13.3
const CUBLAS_STATUS_SUCCESS: c_int = 0;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUDA_R_32F: c_int = 0;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> c_int;

    fn cublasDestroy_v2(handle: *mut c_void) -> c_int;

    fn cublasSetStream_v2(handle: *mut c_void, stream: *mut c_void) -> c_int;

    fn cublasGemmEx(
        handle: *mut c_void,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> c_int;
}

/// An open cuBLAS handle bound to one stream. It is closed on drop.
pub(crate) struct CublasHandle {
    raw: *mut c_void,
    stream: Arc<Stream>,
}

impl CublasHandle {
    /// Opens cuBLAS in the stream's CUDA context and sends its work to that stream.
    pub(crate) fn new(stream: &Arc<Stream>) -> Result<Self, BackendError> {
        stream.device().bind_to_thread().map_err(cuda_error)?;

        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid place for cuBLAS to write the new handle.
        check("cublasCreate", unsafe { cublasCreate_v2(&raw mut raw) })?;
        let handle = Self {
            raw,
            stream: Arc::clone(stream),
        };

        // SAFETY: the handle was just opened, and the stream lives in the same
        // context and is kept alive by the handle.
        check("cublasSetStream", unsafe {
            cublasSetStream_v2(handle.raw, handle.stream.cu_stream().cast())
        })?;
        Ok(handle)
    }

    /// Queues one float32 GEMM on the handle's stream: `C = alpha * op(A) @ B`.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` must be device addresses of float32 buffers in this
    /// handle's context that are large enough for `call`. The buffers must stay
    /// alive until the stream has finished the multiplication.
    pub(crate) unsafe fn gemm_f32(
        &self,
        call: &GemmCall,
        alpha: f32,
        a: usize,
        b: usize,
        c: usize,
    ) -> Result<(), BackendError> {
        self.stream.device().bind_to_thread().map_err(cuda_error)?;

        let transpose_a = if call.transpose_a {
            CUBLAS_OP_T
        } else {
            CUBLAS_OP_N
        };
        let beta = 0.0_f32;

        // SAFETY: the handle is open, cuBLAS copies alpha and beta during the
        // call, and the caller vouches for the three buffers.
        let status = unsafe {
            cublasGemmEx(
                self.raw,
                transpose_a,
                CUBLAS_OP_N,
                call.m,
                call.n,
                call.k,
                (&raw const alpha).cast(),
                a as *const c_void,
                CUDA_R_32F,
                call.lda,
                b as *const c_void,
                CUDA_R_32F,
                call.ldb,
                (&raw const beta).cast(),
                c as *mut c_void,
                CUDA_R_32F,
                call.ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            )
        };
        check("cublasGemmEx", status)
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        let _ = self.stream.device().bind_to_thread();
        // SAFETY: the handle is open and is never used after this call.
        let _ = unsafe { cublasDestroy_v2(self.raw) };
    }
}

fn check(operation: &'static str, status: c_int) -> Result<(), BackendError> {
    if status == CUBLAS_STATUS_SUCCESS {
        return Ok(());
    }
    Err(BackendError::Cublas {
        operation,
        name: status_name(status),
        code: status,
    })
}

fn status_name(status: c_int) -> &'static str {
    match status {
        1 => "CUBLAS_STATUS_NOT_INITIALIZED",
        3 => "CUBLAS_STATUS_ALLOC_FAILED",
        7 => "CUBLAS_STATUS_INVALID_VALUE",
        8 => "CUBLAS_STATUS_ARCH_MISMATCH",
        11 => "CUBLAS_STATUS_MAPPING_ERROR",
        13 => "CUBLAS_STATUS_EXECUTION_FAILED",
        14 => "CUBLAS_STATUS_INTERNAL_ERROR",
        15 => "CUBLAS_STATUS_NOT_SUPPORTED",
        16 => "CUBLAS_STATUS_LICENSE_ERROR",
        _ => "an unknown status",
    }
}

fn cuda_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::Cuda(error.to_string())
}
