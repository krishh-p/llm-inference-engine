# llm-inference-engine

A from-scratch LLM inference runtime and serving systems engine

## Prerequisites

[uv](https://docs.astral.sh/uv/), [rustup](https://rustup.rs/), `make`, and a
[Modal](https://modal.com/) account with billing enabled for GPU use. The
checked-in toolchain file selects the exact Rust version and components.

## Setup

```bash
uv sync --locked
uv run modal setup
```

## Run

```bash
make dev
```

Expect GPU, CUDA, PyTorch, and model-completion details followed by `Smoke test passed`.

Run the manual Qwen3 reference and compare it with Hugging Face:

```bash
make manual
```

Pass another prompt with `make manual PROMPT="2 + 2 ="`. Use
`MAX_NEW_TOKENS=8` the same way.

Run the fixed five-prompt regression set:

```bash
make regression
```

Compare cached decode with recomputing the full sequence for every token:

```bash
make cache-benchmark
```

The benchmark repeats the prompt to 640 tokens so the avoided work is large
enough to show clearly on the GPU.

## Golden fixtures

The Rust tests will use small SafeTensors fixtures generated from the float32
Python reference. They cover RMSNorm, RoPE at both small and large positions,
layer 0 during prefill and decode, and the first greedy tokens.

```bash
make fixtures
```

This writes the fixture files and their manifest to `tests/fixtures/`. The
manifest records the exact model revision, prompt token IDs, positions, tensor
shapes, checksums and expected greedy token IDs. The set is small enough to keep
in the repository. Every model-loading command uses the revision pinned in the
required `model_revision.txt` file, so that file belongs in the same commit.

The RoPE fixture includes the exact `cos` and `sin` tables used for its outputs.
This lets the rotation use a tight tolerance while table generation uses the
looser tolerance recorded in the manifest for large positions.

See [tensor shapes and KV cache memory](docs/shapes-and-kv-cache.md) for the
prefill and decode layouts used by the reference.

## Rust artifact loader

The first Rust slice loads Qwen3 configuration, tokenizer and SafeTensors
metadata without Python. It also validates the committed fixture manifest and
checks that the Rust tokenizer produces the recorded token IDs.

Run the local, host-only checks with:

```bash
make rust-check
```

Inspect a local Hugging Face snapshot containing `config.json`,
`tokenizer.json` and one or more `.safetensors` files with:

```bash
make rust-inspect MODEL_DIR=/path/to/Qwen3-1.7B-Base
```

The loader memory-maps weight files, so inspection reads tensor metadata without
copying the entire checkpoint into memory. `Cargo.lock`, the exact Rust toolchain
and direct dependency versions are committed for reproducible builds.

## Rust CUDA smoke test

The GPU smoke test starts with a small cuTile vector add, then runs the first
real inference kernel: RMSNorm. RMSNorm is checked against PyTorch fixtures at
hidden sizes 128, 2048 and 2560, with both float32 and bfloat16 coverage.

It also checks matrix multiplication through cuBLAS. Tensors are row-major and
cuBLAS is column-major, so the test multiplies small non-square matrices, with
and without a transposed right operand, and compares them with a host
calculation. One more case runs a cuTile kernel, cuBLAS and a cuTile kernel on
the same buffers to show the two libraries share one stream.

```bash
make rust-smoke
```

The GPU code is behind the `cuda` cargo feature, which is off by default, so
`make rust-check` needs no CUDA toolkit. The pinned CUDA image builds the Rust
binary on Modal with the feature on, so this command works from a machine
without a local NVIDIA GPU. It writes the GPU, toolchain,
100-run warm benchmarks, profiler status and kernel verification results to
`results/rust-smoke-*.json`.

Every case here is about ten thousand elements or fewer, so the timings mostly
measure launch overhead. Nsight Compute runs on Modal. Nsight Systems captures a trace but fails
to import it into a report, and the result file records that failure.

## Rust projection check

The first check against the real model loads the pinned Qwen3 weights in Rust,
widens them from bfloat16 to float32, and runs layer 0's query, key and value
projections on the GPU through cuBLAS. Each result is compared with the golden
fixtures, for the five prefill rows and for the single decode row.

```bash
make rust-verify
```

The job reads the weights from the shared Modal Volume and downloads the pinned
snapshot if it is not there yet. It runs the artifact inspection first, then
prints the largest error for each projection. The allowed error is
`atol + rtol * |expected|`, with both values recorded in the fixture manifest.

## Cost and cache
See [Modal pricing](https://modal.com/pricing).

Model files persist across runs in the `llm-inference-engine-hf-cache` Modal Volume; the first run downloads them and Volume storage may incur charges.
