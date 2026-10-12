from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

import modal_common

APP_NAME = "llm-inference-engine-dev"
PROMPT = "The capital of France is"
BYTES_PER_GIB = 1024**3

PROJECT_ROOT = modal_common.find_project_root(__file__)
MODEL_REVISION_PATH, MODEL_REVISION = modal_common.load_model_revision(PROJECT_ROOT)

app = modal.App(APP_NAME)

hf_cache = modal_common.hugging_face_cache()
image = modal_common.make_image(MODEL_REVISION_PATH)


@app.function(
    image=image,
    gpu="L4",
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def smoke_test() -> dict[str, object]:
    import time

    remote_started = time.perf_counter()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    gpu_name = torch.cuda.get_device_name()
    cuda_version = str(torch.version.cuda)
    torch_version = str(torch.__version__)

    # start tracking peak GPU memory before loading the model
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        modal_common.MODEL_NAME,
        revision=MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        modal_common.MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # loading the model onto the GPU can still be in progress. wait for that to finish then stop the timer
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

    # the prompt may still be copying to the GPU, wait for that copy to finish then start the timer
    torch.cuda.synchronize()
    generation_started = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # generating tokens can still be in progress. wait for that to finish then stop the timer
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    generated_tokens = int(outputs.shape[1] - prompt_tokens)
    tokens_per_second = generated_tokens / generation_seconds

    completion = tokenizer.decode(
        outputs[0, prompt_tokens:],
        skip_special_tokens=True,
    )

    gpu_memory_allocated_gib = torch.cuda.memory_allocated() / BYTES_PER_GIB
    gpu_memory_reserved_gib = torch.cuda.memory_reserved() / BYTES_PER_GIB
    gpu_peak_memory_allocated_gib = torch.cuda.max_memory_allocated() / BYTES_PER_GIB

    total_remote_seconds = time.perf_counter() - remote_started

    return {
        "model": modal_common.MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "gpu": gpu_name,
        "cuda": cuda_version,
        "pytorch": torch_version,
        "prompt": PROMPT,
        "completion": completion,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "model_load_seconds": round(model_load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "tokens_per_second": round(tokens_per_second, 2),
        "gpu_memory_allocated_gib": round(gpu_memory_allocated_gib, 3),
        "gpu_memory_reserved_gib": round(gpu_memory_reserved_gib, 3),
        "gpu_peak_memory_allocated_gib": round(
            gpu_peak_memory_allocated_gib,
            3,
        ),
        "total_remote_seconds": round(total_remote_seconds, 3),
    }


@app.local_entrypoint()
def main() -> None:
    result = smoke_test.remote()

    completed_at = datetime.now(UTC)
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
