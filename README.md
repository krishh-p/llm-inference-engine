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

## Cost and cache
See [Modal pricing](https://modal.com/pricing).

Model files persist across runs in the `llm-inference-engine-hf-cache` Modal Volume; the first run downloads them and Volume storage may incur charges.
