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

This is also the Hugging Face baseline. After one warmup run it times the
first token and every decode step, and writes them to `results/smoke-*.json`
under the same field names the Rust engine uses. It loads the model in
bfloat16 by default. Use `make dev DTYPE=float32` to match the Rust engine.
In both engines the load time covers reading the weights and putting them on
the GPU. Loading the tokenizer and setting up CUDA are not counted.

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

Each prompt in `tests/prompts.json` records the greedy token IDs it should
generate, and the run fails if the reference or Hugging Face produces anything
else.

Compare cached decode with recomputing the full sequence for every token:

```bash
make cache-benchmark
```

The benchmark repeats the prompt to 640 tokens so the avoided work is large
enough to show clearly on the GPU.

## Golden fixtures

The Rust tests will use small SafeTensors fixtures generated from the float32
Python reference. They cover RMSNorm, RoPE at both small and large positions,
layer 0 during prefill and decode, and the whole model: the last prompt token
after every layer, the prefill logits, the logits of the first decode step and
the first greedy tokens.

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

The RoPE kernel rotates the fixture's query and key tensors with the reference
`cos` and `sin` tables, at positions from 0 to 32767. The tables are generated
in Rust on the host and compared with the reference tables separately.

The residual add and the RMSNorm that follows it also exist as one fused
kernel. The smoke test checks that it leaves behind the same two tensors as
the separate kernels, then times both versions on a prompt-sized `[5, 2048]`
input and on a `[4096, 2048]` input that is large enough to be limited by
memory rather than by launch overhead.

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

Apart from the `[4096, 2048]` fused case, every input here is about ten
thousand elements or fewer, so those timings mostly measure launch overhead. Nsight Compute runs on Modal. Nsight Systems captures a trace but fails
to import it into a report, and the result file records that failure.

## Rust layer checks

This check runs the real model. It loads the pinned Qwen3 weights in Rust,
widens them from bfloat16 to float32, and runs all of layer 0 on the GPU for
the five-token fixture prompt. It then decodes one more token against the
key and value cache that the prompt left behind. Every stage of both passes is
compared with the golden fixtures: both normalized inputs, the query, key and
value projections, the rotated queries and keys, the attention scores and
probabilities, the attention output, the MLP output and the layer output. The
decode pass also checks the cache before and after the new token is added.

After layer 0, the same command loads the whole model, keeps it on the GPU,
and runs the prompt from its token IDs. Each token's row is copied out of the
embedding table, and the result must match the fixture exactly, since widening
bfloat16 to float32 loses nothing. The rows then go through all 28 layers in
order. The last prompt token is
compared with the fixture after every layer, so a mismatch points to the first
layer that went wrong. Errors grow from about `5e-7` after layer 0 to a few
`1e-4` near the end, where hidden values reach the thousands and float32
rounding steps are larger.

The last token then goes through the final RMSNorm and the LM head. Qwen3-1.7B
ties its LM head to the embedding table, so the head is the
`[151936, 2048]` embedding matrix. The final norm and all 151,936 prefill
logits are compared with the fixture too. Finally, the first generated token
is decoded against the caches the prompt left in every layer, and its logits
are checked the same way.

Last, the command generates greedily from each prompt in `tests/prompts.json`
and requires the exact token IDs recorded for it. The prompt runs once, then
each new token runs on its own against the caches, and the largest logit picks
the next token. Generation ends at the end-of-sequence token or after
`max_new_tokens`. The base model rarely ends a short completion on its own,
so one extra run treats a token it is known to generate as the end-of-sequence
token and checks that generation stops right after it.

Every one of those runs is repeated without caches: for each new token, the
whole sequence so far runs again from the start. Both ways must pick the same
tokens, which shows the caches hold exactly what recomputing would produce.

Each layer's cache is allocated once, with room for the prompt and every new
token, and each token writes only its own row. Attention reads just the filled
rows. The checks give every cache a few spare rows filled with NaN, so reading
a row no token has written shows up as a failure instead of quietly reading
zeros. They also check that a write past the end of a cache is refused before
any layer changes.

Prefill and decode each run on their own workspace, which holds every tensor a
forward pass writes. Prefill's is sized for the prompt and used once. Decode's
is sized for one token and reused for every step, so the token loop allocates
nothing. The attention scores are as wide as the cache and each step uses the
columns for the tokens it holds, the same way it reads the cache. The backend
counts the buffers it allocates, and the checks require that count to stay the
same across every decode step.

```bash
make rust-verify
```

The job reads the weights from the shared Modal Volume and downloads the pinned
snapshot if it is not there yet. It runs the artifact inspection first, then
prints the largest error at each stage. The allowed error is
`atol + rtol * |expected|`, with one pair per stage recorded in the fixture
manifest.

## Rust generation

Generate from any prompt with the Rust engine. Modal is launched from Python,
but there is no Python in the inference path:

```bash
make rust-generate
```

It uses the same `PROMPT` and `MAX_NEW_TOKENS` settings as `make manual`, for
example `make rust-generate PROMPT="2 + 2 =" MAX_NEW_TOKENS=8`. The default run
produces the same 20 tokens as the Python reference and Hugging Face.

The command prints the prompt and generated token IDs and the completion. The
make target only passes the prompt and token limit. The underlying
`qwen3-cli generate` command also takes `--seed` and `--dtype`. Decoding is
greedy, so the seed is recorded but does not change the output, and `float32`
is the only dtype so far. A request that is empty or does not fit in the
32,768-position context fails before the model is loaded.

Each run also writes `results/rust-generate-*.json`. It records the model
load time, one unmeasured warmup run (the first run also compiles every GPU
kernel, which shows up as a slow cold first token), then prefill time, time to
first token, the median and p95 decode step, decode tokens per second, the
host time per token after its logits arrive, and the most device memory in
use. The GPU, driver, CUDA and Rust versions and the git commit go in too.
Median and p95 use the nearest-rank method in both engines.

## Cost and cache
See [Modal pricing](https://modal.com/pricing).

Model files persist across runs in the `llm-inference-engine-hf-cache` Modal Volume; the first run downloads them and Volume storage may incur charges.
