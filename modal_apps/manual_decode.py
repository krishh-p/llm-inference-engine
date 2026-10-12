from __future__ import annotations

import time
from pathlib import Path

import modal

APP_NAME = "llm-inference-engine-manual-decode"
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
GPU_TYPE = "L4"
HF_CACHE_PATH = "/cache/huggingface"

# five test prompts stayed below 1e-4, so leave a little room here
MAX_LOGIT_DIFF = 1e-3

# copy the reference file into the modal image
QWEN3_PATH = Path(__file__).parent.parent / "python" / "reference" / "qwen3.py"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name(
    "llm-inference-engine-hf-cache",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==4.57.6",
        "safetensors==0.8.0",
        "huggingface-hub==0.36.2",
        "accelerate==1.15.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_file(QWEN3_PATH, "/root/qwen3.py")
)


# only available inside the modal image
with image.imports():
    import qwen3
    import torch


def generate_manual(prompt_ids, weights, cfg, eos_ids, max_new_tokens):
    """greedy generation with our reference code."""

    # clear queued gpu work before timing
    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        logits, cache = qwen3.prefill(prompt_ids, weights, cfg)

    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    prefill_logits = logits

    # prefill gives us the first token, then decode handles the rest one at a time
    token = logits.argmax(dim=-1, keepdim=True)
    ids = [token.item()]
    decode_step_seconds = []

    while len(ids) < max_new_tokens and ids[-1] not in eos_ids:
        torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.inference_mode():
            logits, cache = qwen3.decode_step(token, cache, weights, cfg)

        torch.cuda.synchronize()
        decode_step_seconds.append(round(time.perf_counter() - started, 4))

        token = logits.argmax(dim=-1, keepdim=True)
        ids.append(token.item())

    return {
        "ids": ids,
        "prefill_logits": prefill_logits,
        "kv_cache_shape_per_layer": list(cache[0][0].shape),
        "prefill_seconds": round(prefill_seconds, 4),
        "decode_step_seconds": decode_step_seconds,
    }


def generate_hugging_face(model, inputs, pad_id, max_new_tokens):
    """get the expected result from hugging face."""

    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        prefill_logits = model(**inputs).logits[0, -1, :]
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

    return {
        "ids": output[0, prompt_tokens:].tolist(),
        "prefill_logits": prefill_logits,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def manual_inference(prompt: str, max_new_tokens: int) -> dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_ids = inputs["input_ids"][0]

    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    expected = generate_hugging_face(
        model, inputs, tokenizer.eos_token_id, max_new_tokens
    )
    manual = generate_manual(
        prompt_ids, model.state_dict(), model.config, eos_ids, max_new_tokens
    )

    logit_diff = manual["prefill_logits"].float() - expected["prefill_logits"].float()
    max_logit_diff = logit_diff.abs().max().item()

    # fail loudly if we drift from hugging face
    if manual["ids"] != expected["ids"]:
        raise RuntimeError(
            "generated tokens do not match hugging face\n"
            f"expected: {expected['ids']}\n"
            f"manual:   {manual['ids']}"
        )
    if max_logit_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(
            f"prefill logits differ from hugging face by {max_logit_diff}, "
            f"which is above the tolerance of {MAX_LOGIT_DIFF}"
        )

    return {
        "prompt": prompt,
        "hf_completion": tokenizer.decode(expected["ids"], skip_special_tokens=True),
        "manual_completion": tokenizer.decode(manual["ids"], skip_special_tokens=True),
        "prefill_max_logit_diff": max_logit_diff,
        "kv_cache_shape_per_layer": manual["kv_cache_shape_per_layer"],
        "prefill_seconds": manual["prefill_seconds"],
        "decode_step_seconds": manual["decode_step_seconds"],
    }


@app.local_entrypoint()
def main(prompt: str, max_new_tokens: int = 20) -> None:
    result = manual_inference.remote(prompt, max_new_tokens)

    for key, value in result.items():
        print(f"{key}: {value}")
