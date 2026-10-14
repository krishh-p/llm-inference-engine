use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cuda_core::{Device, Event, IntoResult, Stream};
use cuda_kernels::{
    add_f32, add_rms_norm_f32, causal_softmax_f32, merge_heads_f32, rms_norm_f32,
    rms_norm_heads_f32, rope_f32, silu_mul_f32, split_heads_f32,
};
use cutile::prelude::*;

use crate::cublas::CublasHandle;
use crate::matmul::{HeadGemm, check_device, plan_attention_mix, plan_attention_scores, plan_gemm};
use crate::rms_norm::inverse_hidden_size;
use crate::shapes::{check_row_range, dims, expect_shape};
use crate::{BackendError, check_rms_norm_shape};

/// One CUDA device with the stream and cuBLAS handle that every operation uses.
///
/// Each method waits for its work to finish before it returns, so a result can
/// be read, and an input dropped, as soon as the call is over.
pub struct Backend {
    pub(crate) device: Arc<Device>,
    pub(crate) stream: Arc<Stream>,
    cublas: CublasHandle,
    /// Recorded on the stream to wait for the work queued so far.
    finished: Event,
    /// How many device or host buffers this backend has allocated.
    allocations: AtomicUsize,
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
        let finished = device.new_event().map_err(cuda_error)?;

        Ok(Self {
            device,
            stream,
            cublas,
            finished,
            allocations: AtomicUsize::new(0),
        })
    }

    /// How many buffers [`Backend::upload`], [`Backend::zeros`] and
    /// [`Backend::download`] have allocated so far. Comparing two counts shows
    /// whether the work in between allocated anything.
    pub fn allocations(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Device memory in use and in total, in bytes, as the driver reports it.
    ///
    /// This counts everything on the device, including memory that CUDA,
    /// cuBLAS and other processes hold, not just this backend's tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when the driver cannot be asked.
    #[allow(unsafe_code)]
    pub fn memory_in_use(&self) -> Result<(usize, usize), BackendError> {
        self.device.bind_to_thread().map_err(cuda_error)?;
        let mut free = 0;
        let mut total = 0;
        // SAFETY: both are valid places for the driver to write a size.
        unsafe { cuda_core::sys::cuMemGetInfo_v2(&raw mut free, &raw mut total) }
            .result()
            .map_err(cuda_error)?;
        Ok((total.saturating_sub(free), total))
    }

    /// Copies row-major float32 values to the device.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` does not fill `shape` or the copy fails.
    pub fn upload(&self, values: &[f32], shape: &[usize]) -> Result<Tensor<f32>, BackendError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
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
        self.allocations.fetch_add(1, Ordering::Relaxed);
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
        self.allocations.fetch_add(1, Ordering::Relaxed);
        // cuTile's host copy takes ownership, so it is given a device copy.
        let copy = tensor.dup().sync_on(&self.stream).map_err(cutile_error)?;
        copy.to_host_vec()
            .sync_on(&self.stream)
            .map_err(cutile_error)
    }

    /// Copies a tensor into an existing host buffer, in row-major order,
    /// without allocating anything.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is on another device or is not
    /// contiguous, `host` has a different length, or the copy fails.
    #[allow(unsafe_code)]
    pub fn download_into(
        &self,
        tensor: &Tensor<f32>,
        host: &mut [f32],
    ) -> Result<(), BackendError> {
        self.check_tensors(&[("tensor", tensor)])?;
        let elements = tensor.num_bytes() / size_of::<f32>();
        if host.len() != elements {
            return Err(BackendError::OutputLength {
                case: "download_into".to_owned(),
                expected: elements,
                actual: host.len(),
            });
        }

        self.device.bind_to_thread().map_err(cuda_error)?;
        // SAFETY: the tensor was checked to be a contiguous float32 buffer on
        // this device with as many elements as `host` has room for, and both
        // stay borrowed until the stream has been drained below.
        unsafe {
            cuda_core::memcpy_dtoh_async::<f32>(
                host.as_mut_ptr(),
                device_pointer(tensor, 0),
                elements,
                &self.stream,
            )
            .map_err(cuda_error)?;
        }
        self.wait_for_stream()
    }

    /// Copies `rows` rows of one matrix into another of the same width, so
    /// `output[output_row + i] = input[input_row + i]`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device or is not
    /// contiguous, the widths differ, either range runs past its matrix, or
    /// the copy fails.
    #[allow(unsafe_code)]
    pub fn copy_rows(
        &self,
        output: &mut Tensor<f32>,
        output_row: usize,
        input: &Tensor<f32>,
        input_row: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        let [input_rows, width] = dims("copy_rows", "input", input.shape())?;
        let [output_rows, _] = dims("copy_rows", "output", output.shape())?;
        expect_shape("copy_rows", "output", output.shape(), &[output_rows, width])?;
        check_row_range("copy_rows", "input", input_row, rows, input_rows)?;
        check_row_range("copy_rows", "output", output_row, rows, output_rows)?;
        self.check_tensors(&[("input", input), ("output", &*output)])?;

        let row_bytes = width * size_of::<f32>();
        self.device.bind_to_thread().map_err(cuda_error)?;
        // SAFETY: both tensors were checked to be contiguous float32 buffers
        // on this device, both ranges were checked to lie inside them, and
        // they stay borrowed until the stream has been drained below.
        unsafe {
            cuda_core::memcpy_dtod_async::<f32>(
                device_pointer(output, output_row * row_bytes),
                device_pointer(input, input_row * row_bytes),
                rows * width,
                &self.stream,
            )
            .map_err(cuda_error)?;
        }
        self.wait_for_stream()
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

    /// Computes the scaled attention scores of every query head,
    /// `scores[h, :, :keys] = scale * q[h] @ k[h / group, :keys]^T`.
    ///
    /// `q` is `[heads, rows, head_dim]` and `k` is a key cache
    /// `[kv_heads, capacity, head_dim]` holding `keys` tokens. Each key head is
    /// shared by `group` consecutive query heads, which is how grouped-query
    /// attention pairs them. `scores` is `[heads, rows, capacity]`, and only
    /// its first `keys` columns are written.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device or is not
    /// contiguous, the shapes do not agree, or cuBLAS fails.
    pub fn attention_scores(
        &self,
        scores: &mut Tensor<f32>,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        keys: usize,
        scale: f32,
    ) -> Result<(), BackendError> {
        self.check_tensors(&[("q", q), ("k", k), ("scores", &*scores)])?;
        let plan = plan_attention_scores(q.shape(), k.shape(), scores.shape(), keys)?;
        self.head_gemm(&plan, scale, q, k, scores)
    }

    /// Mixes the cached values by the attention probabilities of every query
    /// head, `output[h] = probs[h, :, :keys] @ v[h / group, :keys]`.
    ///
    /// `probs` is `[heads, rows, capacity]`, `v` is a value cache
    /// `[kv_heads, capacity, head_dim]` holding `keys` tokens, and `output` is
    /// `[heads, rows, head_dim]`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device or is not
    /// contiguous, the shapes do not agree, or cuBLAS fails.
    pub fn attention_mix(
        &self,
        output: &mut Tensor<f32>,
        probs: &Tensor<f32>,
        v: &Tensor<f32>,
        keys: usize,
    ) -> Result<(), BackendError> {
        self.check_tensors(&[("probs", probs), ("v", v), ("output", &*output)])?;
        let plan = plan_attention_mix(probs.shape(), v.shape(), output.shape(), keys)?;
        self.head_gemm(&plan, 1.0, probs, v, output)
    }

    /// Computes `output = left + right` for matrices of the same shape.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree, or the kernel fails.
    pub fn add(
        &self,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [rows, cols] = dims("add", "left", left.shape())?;
        expect_shape("add", "right", right.shape(), &[rows, cols])?;
        expect_shape("add", "output", output.shape(), &[rows, cols])?;
        self.check_tensors(&[("left", left), ("right", right), ("output", &*output)])?;

        // elements past the real width are padding and are never stored
        let _ = add_f32(output.partition([1, cols.next_power_of_two()]), left, right)
            .sync_on(&self.stream)
            .map_err(cutile_error)?;
        Ok(())
    }

    /// Computes `output = silu(gate) * up` for matrices of the same shape.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree, or the kernel fails.
    pub fn silu_mul(
        &self,
        output: &mut Tensor<f32>,
        gate: &Tensor<f32>,
        up: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [rows, cols] = dims("silu_mul", "gate", gate.shape())?;
        expect_shape("silu_mul", "up", up.shape(), &[rows, cols])?;
        expect_shape("silu_mul", "output", output.shape(), &[rows, cols])?;
        self.check_tensors(&[("gate", gate), ("up", up), ("output", &*output)])?;

        let _ = silu_mul_f32(output.partition([1, cols.next_power_of_two()]), gate, up)
            .sync_on(&self.stream)
            .map_err(cutile_error)?;
        Ok(())
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

    /// Normalizes every head's vector in a `[heads, rows, head_dim]` tensor
    /// and scales it by `weight`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, a shape is wrong
    /// or unsupported, or the kernel fails.
    pub fn rms_norm_heads(
        &self,
        output: &mut Tensor<f32>,
        input: &Tensor<f32>,
        weight: &Tensor<f32>,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        let [heads, rows, head_dim] = dims("rms_norm_heads", "input", input.shape())?;
        check_rms_norm_shape(heads * rows, head_dim)?;
        expect_shape(
            "rms_norm_heads",
            "output",
            output.shape(),
            &[heads, rows, head_dim],
        )?;
        expect_shape("rms_norm_heads", "weight", weight.shape(), &[head_dim])?;
        self.check_tensors(&[("input", input), ("weight", weight), ("output", &*output)])?;

        let _ = rms_norm_heads_f32(
            output.partition([1, 1, head_dim.next_power_of_two()]),
            input,
            weight,
            inverse_hidden_size(head_dim),
            epsilon,
        )
        .sync_on(&self.stream)
        .map_err(cutile_error)?;
        Ok(())
    }

    /// Computes `sum = left + right` and the `RMSNorm` of that sum with one kernel.
    ///
    /// This gives the same results as [`Backend::add`] followed by
    /// [`Backend::rms_norm`], without reading the sum back from memory.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, a shape is wrong
    /// or unsupported, or the kernel fails.
    pub fn add_rms_norm(
        &self,
        sum: &mut Tensor<f32>,
        output: &mut Tensor<f32>,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
        weight: &Tensor<f32>,
        epsilon: f32,
    ) -> Result<(), BackendError> {
        let [rows, hidden_size] = dims("add_rms_norm", "left", left.shape())?;
        check_rms_norm_shape(rows, hidden_size)?;
        expect_shape("add_rms_norm", "right", right.shape(), &[rows, hidden_size])?;
        expect_shape("add_rms_norm", "sum", sum.shape(), &[rows, hidden_size])?;
        expect_shape(
            "add_rms_norm",
            "output",
            output.shape(),
            &[rows, hidden_size],
        )?;
        expect_shape("add_rms_norm", "weight", weight.shape(), &[hidden_size])?;
        self.check_tensors(&[
            ("left", left),
            ("right", right),
            ("weight", weight),
            ("sum", &*sum),
            ("output", &*output),
        ])?;

        let tile = [1, hidden_size.next_power_of_two()];
        let _ = add_rms_norm_f32(
            sum.partition(tile),
            output.partition(tile),
            left,
            right,
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

    /// Splits `[rows, heads * head_dim]` into `[heads, rows, head_dim]`, so
    /// each head's vectors sit together.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree, or the kernel fails.
    pub fn split_heads(
        &self,
        output: &mut Tensor<f32>,
        input: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [heads, rows, head_dim] = dims("split_heads", "output", output.shape())?;
        expect_shape(
            "split_heads",
            "input",
            input.shape(),
            &[rows, heads * head_dim],
        )?;
        self.check_tensors(&[("input", input), ("output", &*output)])?;

        let _ = split_heads_f32(output.partition([1, 1, head_dim]), input)
            .sync_on(&self.stream)
            .map_err(cutile_error)?;
        Ok(())
    }

    /// Merges `[heads, rows, head_dim]` back into `[rows, heads * head_dim]`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device, the shapes do not
    /// agree, or the kernel fails.
    pub fn merge_heads(
        &self,
        output: &mut Tensor<f32>,
        input: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [heads, rows, head_dim] = dims("merge_heads", "input", input.shape())?;
        expect_shape(
            "merge_heads",
            "output",
            output.shape(),
            &[rows, heads * head_dim],
        )?;
        self.check_tensors(&[("input", input), ("output", &*output)])?;

        let _ = merge_heads_f32(output.partition([1, head_dim]), input)
            .sync_on(&self.stream)
            .map_err(cutile_error)?;
        Ok(())
    }

    /// Copies every head of a `[heads, rows, width]` tensor into a
    /// `[heads, capacity, width]` tensor, starting at row `first_row`, so
    /// `output[h, first_row + r] = input[h, r]`.
    ///
    /// This adds new keys or values to a KV cache without touching the rows
    /// already in it.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is on another device or is not
    /// contiguous, the heads or widths differ, the rows run past the end of
    /// `output`, or a copy fails.
    #[allow(unsafe_code)]
    pub fn write_rows(
        &self,
        output: &mut Tensor<f32>,
        first_row: usize,
        input: &Tensor<f32>,
    ) -> Result<(), BackendError> {
        let [heads, rows, width] = dims("write_rows", "input", input.shape())?;
        let [_, capacity, _] = dims("write_rows", "output", output.shape())?;
        expect_shape(
            "write_rows",
            "output",
            output.shape(),
            &[heads, capacity, width],
        )?;
        check_row_range("write_rows", "output", first_row, rows, capacity)?;
        self.check_tensors(&[("input", input), ("output", &*output)])?;

        let row_bytes = width * size_of::<f32>();
        self.device.bind_to_thread().map_err(cuda_error)?;
        // output heads are `capacity` rows apart, so each head is its own copy
        for head in 0..heads {
            let output_offset = (head * capacity + first_row) * row_bytes;
            let input_offset = head * rows * row_bytes;

            // SAFETY: both tensors were checked to be contiguous float32
            // buffers on this device, the row range was checked to lie inside
            // every output head, and they stay borrowed until the stream has
            // been drained below.
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    device_pointer(output, output_offset),
                    device_pointer(input, input_offset),
                    rows * width,
                    &self.stream,
                )
                .map_err(cuda_error)?;
            }
        }
        self.wait_for_stream()
    }

    /// Turns `[heads, rows, cols]` attention scores into probabilities.
    ///
    /// Row `r` belongs to the token at `first_position + r` and column `c` to
    /// the token at position `c`. A token sees itself and earlier tokens only.
    /// Scores can be wider than the tokens they hold (sized for the whole KV
    /// cache, say). Columns past a row's token just come out as 0.
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
        // the last row is the newest token, so there must be a column for it
        if cols < first_position + rows {
            return Err(BackendError::Shape {
                operation: "causal_softmax",
                name: "scores",
                expected: vec![heads, rows, first_position + rows],
                actual: scores.shape().to_vec(),
            });
        }
        expect_shape(
            "causal_softmax",
            "output",
            output.shape(),
            &[heads, rows, cols],
        )?;
        self.check_tensors(&[("scores", scores), ("output", &*output)])?;
        let first_position = i32::try_from(first_position).map_err(|_| BackendError::Shape {
            operation: "causal_softmax",
            name: "scores",
            expected: vec![heads, rows, first_position + rows],
            actual: scores.shape().to_vec(),
        })?;

        // the causal mask also hides unused cache columns and the padding
        // up to the tile width
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

    /// Runs one planned GEMM per head: `output[h] = scale * left[h] @ right[h / group]`,
    /// with `right` transposed when the plan says so.
    #[allow(unsafe_code)]
    fn head_gemm(
        &self,
        plan: &HeadGemm,
        scale: f32,
        left: &Tensor<f32>,
        right: &Tensor<f32>,
        output: &mut Tensor<f32>,
    ) -> Result<(), BackendError> {
        for head in 0..plan.heads {
            let left_offset = head * plan.left_elements * size_of::<f32>();
            let right_offset = (head / plan.group) * plan.right_elements * size_of::<f32>();
            let output_offset = head * plan.output_elements * size_of::<f32>();

            // SAFETY: the caller checked that the three tensors are contiguous
            // float32 buffers on this device, the plan checked that every head
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
        self.finished.record(&self.stream).map_err(cuda_error)?;
        self.finished.synchronize().map_err(cuda_error)
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

/// The device address `offset_bytes` into a tensor, as the CUDA driver takes it.
fn device_pointer(tensor: &Tensor<f32>, offset_bytes: usize) -> cuda_core::sys::CUdeviceptr {
    (device_address(tensor) + offset_bytes)
        .try_into()
        .expect("CUDA device addresses fit in CUdeviceptr")
}

fn cuda_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::Cuda(error.to_string())
}

fn cutile_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::Cutile(error.to_string())
}
