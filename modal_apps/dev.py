from __future__ import annotations

import itertools
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

import modal_common

APP_NAME = "llm-inference-engine-dev"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20
WARMUP_RUNS = 1
BYTES_PER_GIB = 1024**3

PROJECT_ROOT = modal_common.find_project_root(__file__)
MODEL_REVISION_PATH, MODEL_REVISION = modal_common.load_model_revision(PROJECT_ROOT)

app = modal.App(APP_NAME)

hf_cache = modal_common.hugging_face_cache()
image = modal_common.make_image(MODEL_REVISION_PATH)


class TokenTimer:
    """records when generate hands over each new token"""

    def __init__(self):
        self.prompt_seen = False
        self.token_times = []

    def put(self, value):
        # the first call is the prompt, every later one is a new token
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.token_times.append(time.perf_counter())

    def end(self):
        pass


def percentile(values, percent):
    # nearest rank, the same way the rust engine reports it
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent * len(ordered) / 100))
    return ordered[rank - 1]


def milliseconds(seconds):
    # none when there were no decode steps, e.g. the first token was eos
    return None if seconds is None else round(seconds * 1000, 3)


@app.function(
    image=image,
    gpu="L4",
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def smoke_test(dtype: str) -> dict[str, object]:
    remote_started = time.perf_counter()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    torch_dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype not in torch_dtypes:
        raise ValueError(f"dtype must be one of {list(torch_dtypes)}, not {dtype!r}")

    gpu_name = torch.cuda.get_device_name()
    cuda_version = str(torch.version.cuda)
    torch_version = str(torch.__version__)

    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        modal_common.MODEL_NAME,
        revision=MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        modal_common.MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch_dtypes[dtype],
        device_map="auto",
    )
    model.eval()

    # gpu work is queued, so wait for it to finish before reading the clock
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"the model was loaded on {model_device}, not CUDA")

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    ).to("cuda")

    prompt_tokens = int(inputs["input_ids"].shape[1])

    def generate(streamer=None):
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

    # the first run picks kernels and fills caches, so it isn't measured
    for _ in range(WARMUP_RUNS):
        generate()

    timer = TokenTimer()
    torch.cuda.synchronize()
    generation_started = time.perf_counter()
    outputs = generate(timer)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    generated_tokens = int(outputs.shape[1] - prompt_tokens)
    tokens_per_second = generated_tokens / generation_seconds

    # one gap between tokens per decode step
    times = timer.token_times
    time_to_first_token_seconds = times[0] - generation_started
    decode_step_seconds = [b - a for a, b in itertools.pairwise(times)]
    decode_tokens_per_second = (
        len(decode_step_seconds) / sum(decode_step_seconds)
        if decode_step_seconds
        else None
    )

    completion = tokenizer.decode(
        outputs[0, prompt_tokens:],
        skip_special_tokens=True,
    )

    gpu_memory_allocated_gib = torch.cuda.memory_allocated() / BYTES_PER_GIB
    gpu_memory_reserved_gib = torch.cuda.memory_reserved() / BYTES_PER_GIB
    gpu_peak_memory_allocated_gib = torch.cuda.max_memory_allocated() / BYTES_PER_GIB

    # what the driver sees in use, the same number the rust engine reports
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gpu_memory_used_gib = (total_bytes - free_bytes) / BYTES_PER_GIB

    total_remote_seconds = time.perf_counter() - remote_started

    return {
        "engine": "pytorch",
        "model": modal_common.MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "gpu": gpu_name,
        "cuda": cuda_version,
        "pytorch": torch_version,
        "dtype": dtype,
        "prompt": PROMPT,
        "completion": completion,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "warmup_runs": WARMUP_RUNS,
        "model_load_seconds": round(model_load_seconds, 3),
        "time_to_first_token_seconds": round(time_to_first_token_seconds, 4),
        "decode_step_median_ms": milliseconds(percentile(decode_step_seconds, 50)),
        "decode_step_p95_ms": milliseconds(percentile(decode_step_seconds, 95)),
        "decode_tokens_per_second": decode_tokens_per_second,
        "generation_seconds": round(generation_seconds, 3),
        "tokens_per_second": round(tokens_per_second, 2),
        "gpu_memory_used_gib": round(gpu_memory_used_gib, 3),
        "gpu_memory_total_gib": round(total_bytes / BYTES_PER_GIB, 3),
        "gpu_memory_allocated_gib": round(gpu_memory_allocated_gib, 3),
        "gpu_memory_reserved_gib": round(gpu_memory_reserved_gib, 3),
        "gpu_peak_memory_allocated_gib": round(
            gpu_peak_memory_allocated_gib,
            3,
        ),
        "total_remote_seconds": round(total_remote_seconds, 3),
    }


def git(*args):
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@app.local_entrypoint()
def main(dtype: str = "bfloat16") -> None:
    result = smoke_test.remote(dtype)

    completed_at = datetime.now(UTC)
    result["git_commit"] = git("rev-parse", "HEAD")
    result["git_dirty"] = bool(git("status", "--porcelain"))
    result["completed_at_utc"] = completed_at.isoformat()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = completed_at.strftime("smoke-%Y%m%dT%H%M%SZ.json")
    output_path = results_dir / filename

    output_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("smoke test passed")
    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"result_json: {output_path}")
