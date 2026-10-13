from __future__ import annotations

import json
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "llm-inference-engine-rust-smoke"
CUDA_IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
REMOTE_ROOT = "/workspace"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# the toolchain file is the single place the Rust version is pinned
TOOLCHAIN_PATH = PROJECT_ROOT / "rust-toolchain.toml"
if not TOOLCHAIN_PATH.is_file():
    TOOLCHAIN_PATH = Path(REMOTE_ROOT) / "rust-toolchain.toml"
RUST_VERSION = tomllib.loads(TOOLCHAIN_PATH.read_text(encoding="utf-8"))["toolchain"][
    "channel"
]

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "libclang-dev",
        "pkg-config",
    )
    .env(
        {
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUTILE_TILEIRAS_PATH": "/usr/local/cuda/bin/tileiras",
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        f"sh -s -- -y --profile minimal --default-toolchain {RUST_VERSION}"
    )
    .add_local_file(PROJECT_ROOT / "Cargo.toml", f"{REMOTE_ROOT}/Cargo.toml", copy=True)
    .add_local_file(PROJECT_ROOT / "Cargo.lock", f"{REMOTE_ROOT}/Cargo.lock", copy=True)
    .add_local_file(
        PROJECT_ROOT / "rust-toolchain.toml",
        f"{REMOTE_ROOT}/rust-toolchain.toml",
        copy=True,
    )
    .add_local_dir(PROJECT_ROOT / "crates", f"{REMOTE_ROOT}/crates", copy=True)
    .run_commands(
        f"cd {REMOTE_ROOT} && cargo clippy --locked --release "
        "-p cuda-backend -p cuda-kernels --all-targets -- -D warnings && "
        f"cd {REMOTE_ROOT} && cargo build --locked --release "
        "-p cuda-backend --bin cuda-smoke"
    )
)


def run_command(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@app.function(image=image, gpu="L4", timeout=600)
def run_on_gpu() -> dict[str, object]:
    binary = f"{REMOTE_ROOT}/target/release/cuda-smoke"
    completed = subprocess.run(
        [binary],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Rust CUDA smoke test failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    result = json.loads(completed.stdout)
    result["environment"] = {
        "cuda_image": CUDA_IMAGE,
        "rustc": run_command("rustc", "--version"),
        "cargo": run_command("cargo", "--version"),
        "nvcc": run_command("nvcc", "--version").splitlines()[-1],
        "tileiras": run_command("tileiras", "--version"),
    }
    return result


def git_metadata() -> tuple[str, bool]:
    commit = run_command("git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD")
    status = run_command("git", "-C", str(PROJECT_ROOT), "status", "--porcelain")
    return commit, bool(status)


@app.local_entrypoint()
def main() -> None:
    result = run_on_gpu.remote()
    commit, dirty = git_metadata()
    completed_at = datetime.now(UTC)

    result["git_commit"] = commit
    result["git_dirty"] = dirty
    result["completed_at_utc"] = completed_at.isoformat()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / completed_at.strftime("rust-smoke-%Y%m%dT%H%M%SZ.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    vector_add = result["vector_add"]
    gpu = result["gpu"]
    print("Rust CUDA smoke test passed")
    print(f"gpu: {gpu['name']} ({gpu['compute_capability']})")
    print(f"first_run_ms: {vector_add['first_run_ms']:.3f}")
    print(f"warm_run_ms: {vector_add['warm_run_ms']:.3f}")
    print(f"max_abs_error: {vector_add['max_abs_error']}")
    print(f"result_json: {output_path}")
