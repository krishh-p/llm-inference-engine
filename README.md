# llm-inference-engine

A from-scratch LLM inference engine in Rust for Qwen3-1.7B. It loads the
model's Hugging Face weights, runs prefill and cached token-by-token decode on
an NVIDIA GPU, and generates text without Python or PyTorch in the inference
path.

Rust handles the model loading, tensor shapes, KV cache and generation loop.
The GPU work uses custom cuTile kernels written in Rust, including RMSNorm, a
fused residual-add + RMSNorm, RoPE, causal softmax and gated SiLU. cuBLAS
handles the matrix multiplications.

A float32 Python reference exports golden tensors at the important boundaries.
The Rust engine is checked against them on the GPU, including every layer's
output, the final logits, and the layer-0 KV cache before and after decode.
Greedy output matches Hugging Face token for token on five fixed prompts, and
cached decode matches recomputing the full sequence from scratch.

Right now the engine runs in float32 at batch size 1 with greedy decoding.
bfloat16, fused attention and batched serving are coming soon 👀

## Setup

You need [uv](https://docs.astral.sh/uv/),
[rustup](https://rustup.rs/), `make`, and a
[Modal](https://modal.com/) account with GPU access.

The Rust toolchain, Python packages, model revision and direct dependencies are
all pinned in the repo.

```bash
uv sync --locked
uv run modal setup
```

## Python reference

The Python code gives the Rust engine something trustworthy to compare against.
It can also run the same model through Hugging Face as a second check.

Run the Hugging Face baseline:

```bash
make dev
```

It uses bfloat16 by default. Use `make dev DTYPE=float32` for a direct
comparison with the Rust engine. After one warmup, it records load, first-token,
decode and memory numbers in `results/smoke-*.json`.

Run the handwritten reference and compare its logits and tokens with Hugging
Face:

```bash
make manual
```

You can pass a different prompt and token limit:

```bash
make manual PROMPT="2 + 2 =" MAX_NEW_TOKENS=8
```

There is also a fixed five-prompt regression set:

```bash
make regression
```

The expected token IDs live in `tests/prompts.json`. The run fails if either
the reference or Hugging Face generates different tokens.

To see why the KV cache matters, time cached decode against rerunning the whole
sequence for every token:

```bash
make cache-benchmark
```

## Golden fixtures

The float32 Python reference saves intermediate tensors as small SafeTensors
files:

```bash
make fixtures
```

They cover RMSNorm, RoPE, layer 0 prefill and decode, the KV cache, every
layer's final-token output, final logits and greedy tokens. The manifest in
`tests/fixtures/` records the shapes, checksums, tolerances, prompt IDs and exact
model revision used to create them.

The files are small enough to keep in git. RoPE also stores its exact `cos` and
`sin` tables, so the rotation and the table math can be checked separately.

The prefill and decode layouts are written out in
[docs/shapes-and-kv-cache.md](docs/shapes-and-kv-cache.md).

## Rust checks

Start with the local checks:

```bash
make rust-check
```

This runs formatting, clippy with warnings denied, and the host-side tests. CUDA
is behind a cargo feature, so none of this needs a local NVIDIA GPU.

You can also inspect a downloaded Qwen snapshot without loading the full model
into memory:

```bash
make rust-inspect MODEL_DIR=/path/to/Qwen3-1.7B-Base
```

Run the kernel smoke tests on a Modal L4:

```bash
make rust-smoke
```

This checks the cuTile kernels, cuBLAS matrix layouts and the fused
residual-add + RMSNorm against known outputs. It also times warm runs and saves
the results to `results/rust-smoke-*.json`.

The main correctness check runs the actual model:

```bash
make rust-verify
```

It checks every boundary in layer 0 for both prefill and cached decode,
including the KV cache before and after the new token is added. It then checks
the output after each of the 28 layers, the final norm and the full vocabulary
logits. Errors grow from about `5e-7` after layer 0 to a few `1e-4` near the
end, where hidden values reach the thousands and float32's rounding steps are
larger. Finally, it generates from all five regression prompts both with and
without the cache. The token IDs must match each other and the recorded Python
and Hugging Face outputs.

The cache is allocated once and decode reuses one workspace. The verifier also
checks that the token loop does not allocate any new engine buffers. If a value
falls outside its fixture tolerance, the output names the first boundary that
went wrong.

## Rust generation

Run the engine end to end:

```bash
make rust-generate
```

Pass a prompt and token limit the same way as the Python reference:

```bash
make rust-generate PROMPT="2 + 2 =" MAX_NEW_TOKENS=8
```

The Modal command is launched with Python, but inference itself is done by the
standalone Rust binary. It prints the token IDs and completion, stops at EOS or
the token limit, and rejects empty or oversized requests before loading the
model.

Each run does one warmup and writes `results/rust-generate-*.json` with model
load, prefill, first-token, decode and memory numbers. It also records the GPU,
CUDA, Rust version, model revision and git commit so a run can be traced back to
the code that produced it.

## Rust decode benchmark

```bash
make rust-bench
```

This times steady-state decode steps with 128, 512 and 2048 tokens already in
the cache, filled by repeating a short prompt. For each length, a prefill and
16 decode steps first run on a cache that is then thrown away, so every kernel
is compiled. A fresh cache is then filled and 32 steps are timed, starting from
exactly that many tokens. It writes `results/rust-bench-*.json` with the min,
median, p95 and max step, decode tokens per second, prefill time, and every
step's time, plus the same environment and git details as above.

It also counts each step's cuTile kernel launches, cuBLAS calls, and the times
the host waits for the GPU to finish. The backend is still synchronous, so every
cuTile launch and each group of cuBLAS calls is followed by a host wait.

The warmup is 16 steps because cuTile compiles a kernel again whenever an
integer argument's largest power-of-two divisor changes, up to 16. The softmax
takes the token's position, so the first steps after a prefill each hit one of
those compiles. Any 16 positions in a row cover every version.

## Cost and cache

The first run downloads the model into the shared
`llm-inference-engine-hf-cache` Modal Volume. Later runs reuse it. GPU time and
stored model files may cost money; see [Modal pricing](https://modal.com/pricing).
