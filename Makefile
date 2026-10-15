.PHONY: dev manual fixtures regression cache-benchmark rust-check rust-inspect rust-smoke rust-verify rust-generate rust-bench

PROMPT ?= The capital of France is
MAX_NEW_TOKENS ?= 20
DTYPE ?= bfloat16

dev:
	uv run modal run modal_apps/dev.py --dtype $(DTYPE)

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

rust-smoke:
	uv run modal run modal_apps/rust_smoke.py::main

rust-verify:
	uv run modal run modal_apps/rust_smoke.py::run_verify

rust-generate:
	uv run modal run modal_apps/rust_smoke.py::run_generate --prompt "$(PROMPT)" --max-new-tokens $(MAX_NEW_TOKENS)

rust-bench:
	uv run modal run modal_apps/rust_smoke.py::run_bench
