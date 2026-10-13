.PHONY: dev manual fixtures regression cache-benchmark rust-check rust-inspect

PROMPT ?= The capital of France is
MAX_NEW_TOKENS ?= 20

dev:
	uv run modal run modal_apps/dev.py

manual:
	uv run modal run modal_apps/manual_decode.py::main --prompt "$(PROMPT)" --max-new-tokens $(MAX_NEW_TOKENS)

fixtures:
	uv run modal run python/reference/export_fixtures.py::main

regression:
	uv run modal run modal_apps/manual_decode.py::run_regression

cache-benchmark:
	uv run modal run modal_apps/manual_decode.py::run_cache_benchmark

rust-check:
	cargo fmt --check
	cargo clippy --locked --workspace --all-targets -- -D warnings
	cargo test --locked --workspace

rust-inspect:
	@test -n "$(MODEL_DIR)" || (echo "set MODEL_DIR to a local Qwen3 snapshot" && exit 1)
	cargo run --locked -p qwen3-cli -- inspect --model-dir "$(MODEL_DIR)"
