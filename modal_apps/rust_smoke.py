from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import modal

import modal_common

APP_NAME = "llm-inference-engine-rust-smoke"
CUDA_IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
REMOTE_ROOT = "/workspace"
REMOTE_FIXTURE = "/fixtures/rms_norm.safetensors"
BENCHMARK_WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# the toolchain file is the single place the Rust version is pinned
TOOLCHAIN_PATH = PROJECT_ROOT / "rust-toolchain.toml"
if not TOOLCHAIN_PATH.is_file():
    TOOLCHAIN_PATH = Path(REMOTE_ROOT) / "rust-toolchain.toml"
RUST_VERSION = tomllib.loads(TOOLCHAIN_PATH.read_text(encoding="utf-8"))["toolchain"][
    "channel"
]

app = modal.App(APP_NAME)

hf_cache = modal_common.hugging_face_cache()

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
    .uv_pip_install("huggingface-hub==0.36.2")
    .env(
        {
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUTILE_TILEIRAS_PATH": "/usr/local/cuda/bin/tileiras",
            "HF_HOME": modal_common.HF_CACHE_PATH,
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
    .add_local_file(
        PROJECT_ROOT / "model_revision.txt",
        f"{REMOTE_ROOT}/model_revision.txt",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "modal_common.py", "/root/modal_common.py", copy=True
    )
    .add_local_dir(PROJECT_ROOT / "tests", f"{REMOTE_ROOT}/tests", copy=True)
    .add_local_dir(PROJECT_ROOT / "crates", f"{REMOTE_ROOT}/crates", copy=True)
    .run_commands(
        f"cd {REMOTE_ROOT} && cargo clippy --locked --release --workspace "
        "--features qwen3-cli/cuda --all-targets -- -D warnings && "
        f"cd {REMOTE_ROOT} && cargo build --locked --release --workspace "
        "--features qwen3-cli/cuda"
    )
)

pytorch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.0", "safetensors==0.8.0")
    .add_local_file(
        PROJECT_ROOT / "rust-toolchain.toml",
        f"{REMOTE_ROOT}/rust-toolchain.toml",
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "tests/fixtures/rms_norm.safetensors",
        REMOTE_FIXTURE,
        copy=True,
    )
    .add_local_file(
        PROJECT_ROOT / "modal_common.py", "/root/modal_common.py", copy=True
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


def command_summary(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return output[-6000:]


def profile_rust(binary: str) -> dict[str, object]:
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    result = {
        "nsight_systems": {"available": nsys is not None},
        "nsight_compute": {"available": ncu is not None},
    }

    profiler_timed_out = False
    if nsys is not None:
        report = "/tmp/rust-smoke.nsys-rep"
        try:
            completed = subprocess.run(
                [
                    nsys,
                    "profile",
                    "--trace=cuda",
                    "--sample=none",
                    "--force-overwrite=true",
                    "--output=/tmp/rust-smoke",
                    binary,
                ],
                check=False,
                capture_output=True,
                cwd=REMOTE_ROOT,
                text=True,
                timeout=60,
            )
            report_exists = Path(report).is_file()
            succeeded = completed.returncode == 0 and report_exists
            details = {
                "available": True,
                "succeeded": succeeded,
                "command_output": command_summary(completed),
            }
            if completed.returncode == 0 and not report_exists:
                details["error"] = (
                    "capture completed, but the Nsight importer did not create "
                    "an .nsys-rep file on Modal"
                )
            if succeeded:
                stats = subprocess.run(
                    [
                        nsys,
                        "stats",
                        "--report=cuda_gpu_kern_sum,cuda_api_sum",
                        "--format=csv",
                        report,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                details["stats"] = command_summary(stats)
            result["nsight_systems"] = details
        except subprocess.TimeoutExpired:
            profiler_timed_out = True
            result["nsight_systems"] = {
                "available": True,
                "succeeded": False,
                "error": "timed out after 60 seconds on Modal",
            }

    if ncu is not None and not profiler_timed_out:
        try:
            completed = subprocess.run(
                [
                    ncu,
                    "--set=basic",
                    "--target-processes=all",
                    "--kernel-name=regex:rms_norm",
                    "--launch-count=1",
                    "--clock-control=none",
                    "--csv",
                    binary,
                ],
                check=False,
                capture_output=True,
                cwd=REMOTE_ROOT,
                text=True,
                timeout=60,
            )
            result["nsight_compute"] = {
                "available": True,
                "succeeded": completed.returncode == 0,
                "summary": command_summary(completed),
            }
        except subprocess.TimeoutExpired:
            result["nsight_compute"] = {
                "available": True,
                "succeeded": False,
                "error": "timed out after 60 seconds on Modal",
            }
    elif ncu is not None:
        result["nsight_compute"] = {
            "available": True,
            "succeeded": False,
            "error": "skipped after Nsight Systems timed out",
        }

    return result


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
    result["profilers"] = profile_rust(binary)
    return result


@app.function(
    image=image,
    gpu="L4",
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=1200,
)
def verify_on_gpu() -> str:
    from huggingface_hub import snapshot_download

    revision_path = Path(REMOTE_ROOT) / "model_revision.txt"
    revision = revision_path.read_text(encoding="utf-8").strip()

    # reuses the snapshot in the volume, and downloads it if it is not there yet
    model_dir = snapshot_download(
        modal_common.MODEL_NAME,
        revision=revision,
        allow_patterns=["config.json", "tokenizer.json", "*.safetensors"],
    )

    binary = f"{REMOTE_ROOT}/target/release/qwen3-cli"
    outputs = []
    for command in ("inspect", "verify"):
        completed = subprocess.run(
            [binary, command, "--model-dir", model_dir],
            check=False,
            capture_output=True,
            cwd=REMOTE_ROOT,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"qwen3-cli {command} failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        outputs.append(completed.stdout.strip())

    return "\n".join(outputs)


@app.function(image=pytorch_image, gpu="L4", timeout=600)
def benchmark_pytorch() -> dict[str, object]:
    import torch
    from safetensors.torch import load_file

    tensors = load_file(REMOTE_FIXTURE, device="cuda")
    input_tensor = tensors["input"]
    weight = tensors["weight"]
    expected = tensors["output"]
    epsilon = float(tensors["eps"].item())

    def rms_norm(x, scale):
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + epsilon
        )
        return normalized.to(x.dtype) * scale

    def measure(name, operation):
        for _ in range(BENCHMARK_WARMUP_ITERATIONS):
            actual = operation(input_tensor, weight)
        torch.cuda.synchronize()

        timings = []
        for _ in range(BENCHMARK_ITERATIONS):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            actual = operation(input_tensor, weight)
            finished.record()
            finished.synchronize()
            timings.append(started.elapsed_time(finished))

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            actual = operation(input_tensor, weight)
            torch.cuda.synchronize()
        launches = sum(
            getattr(event.device_type, "name", "") == "CUDA"
            for event in profiler.events()
        )

        timings.sort()
        median_ms = timings[(len(timings) - 1) * 50 // 100]
        logical_bytes = input_tensor.numel() * input_tensor.element_size() * 3
        max_abs_error = (actual.float() - expected.float()).abs().max().item()
        return {
            "name": name,
            "warmup_iterations": BENCHMARK_WARMUP_ITERATIONS,
            "measured_iterations": BENCHMARK_ITERATIONS,
            "min_ms": timings[0],
            "median_ms": median_ms,
            "p95_ms": timings[(len(timings) - 1) * 95 // 100],
            "effective_bandwidth_gbps": logical_bytes / (median_ms / 1000) / 1e9,
            "launches_per_iteration": launches,
            "max_abs_error": max_abs_error,
        }

    with torch.inference_mode():
        eager = measure("pytorch_eager", rms_norm)
        try:
            compiled_operation = torch.compile(
                rms_norm,
                fullgraph=True,
                mode="reduce-overhead",
            )
            compiled = measure("pytorch_compiled", compiled_operation)
            compiled["available"] = True
        except Exception as error:  # noqa: BLE001
            compiled = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }

    return {
        "case": "layer_input_2048_f32",
        "dtype": "float32",
        "rows": int(input_tensor.shape[0]),
        "hidden_size": int(input_tensor.shape[1]),
        "pytorch_version": str(torch.__version__),
        "gpu": str(torch.cuda.get_device_name()),
        "eager": eager,
        "compiled": compiled,
    }


def git_metadata() -> tuple[str, bool]:
    commit = run_command("git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD")
    status = run_command("git", "-C", str(PROJECT_ROOT), "status", "--porcelain")
    return commit, bool(status)


@app.local_entrypoint()
def main() -> None:
    result = run_on_gpu.remote()
    result["pytorch_benchmark"] = benchmark_pytorch.remote()
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
    rms_norm = result["rms_norm"]
    gpu = result["gpu"]
    print("Rust CUDA smoke test passed")
    print(f"gpu: {gpu['name']} ({gpu['compute_capability']})")
    print(f"first_run_ms: {vector_add['first_run_ms']:.3f}")
    print(f"warm_run_ms: {vector_add['warm_run_ms']:.3f}")
    print(f"max_abs_error: {vector_add['max_abs_error']}")
    for case in rms_norm["cases"]:
        benchmark = case["benchmark"]
        print(
            f"{case['name']}: max_abs_error={case['max_abs_error']}, "
            f"median_ms={benchmark['median_ms']:.4f}, "
            f"bandwidth_gbps={benchmark['effective_bandwidth_gbps']:.2f}"
        )
    tables = result["rope"]["tables"]
    print(
        f"rope_tables: cos_max_abs_error={tables['cos_max_abs_error']}, "
        f"sin_max_abs_error={tables['sin_max_abs_error']}"
    )
    for case in result["rope"]["cases"]:
        print(f"{case['name']}: max_abs_error={case['max_abs_error']}")
    for case in result["matmul"]["cases"]:
        print(f"matmul_{case['name']}: max_abs_error={case['max_abs_error']}")
    pytorch = result["pytorch_benchmark"]
    print(f"pytorch_eager_median_ms: {pytorch['eager']['median_ms']:.4f}")
    if pytorch["compiled"]["available"]:
        print(f"pytorch_compiled_median_ms: {pytorch['compiled']['median_ms']:.4f}")
    else:
        print("pytorch_compiled: unavailable")
    print(f"result_json: {output_path}")


@app.local_entrypoint()
def run_verify() -> None:
    print(verify_on_gpu.remote())
