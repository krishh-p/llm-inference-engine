# llm-inference-engine

A from-scratch LLM inference runtime and serving systems engine

## Prerequisites

[uv](https://docs.astral.sh/uv/), `make`, and a [Modal](https://modal.com/) account with billing enabled for GPU use.

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

## Cost and cache
See [Modal pricing](https://modal.com/pricing).

Model files persist across runs in the `llm-inference-engine-hf-cache` Modal Volume; the first run downloads them and Volume storage may incur charges.
