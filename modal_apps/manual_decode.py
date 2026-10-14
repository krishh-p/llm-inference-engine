from __future__ import annotations

import json
import time
from pathlib import Path

import modal

import modal_common

APP_NAME = "llm-inference-engine-manual-decode"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20
CACHE_BENCHMARK_PROMPT_REPETITIONS = 128

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = modal_common.find_project_root(__file__)
MODEL_REVISION_PATH, MODEL_REVISION = modal_common.load_model_revision(PROJECT_ROOT)
PROMPT_SET_PATH = PROJECT_ROOT / "tests" / "prompts.json"
QWEN3_PATH = PROJECT_ROOT / "python" / "reference" / "qwen3.py"
if not QWEN3_PATH.is_file():
    QWEN3_PATH = SOURCE_DIR / "qwen3.py"

app = modal.App(APP_NAME)

hf_cache = modal_common.hugging_face_cache()
image = modal_common.make_image(
    MODEL_REVISION_PATH,
    local_files=((QWEN3_PATH, "/root/qwen3.py"),),
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


def generate_cached(prompt_ids, weights, cfg, eos_ids, max_new_tokens):
    logits, cache = qwen3.prefill(prompt_ids, weights, cfg)
    ids = []

    while len(ids) < max_new_tokens:
        token = logits.argmax(dim=-1, keepdim=True)
        ids.append(token.item())
        if token.item() in eos_ids:
            break
        logits, cache = qwen3.decode_step(token, cache, weights, cfg)

    return ids


def generate_uncached(prompt_ids, weights, cfg, eos_ids, max_new_tokens):
    tokens = prompt_ids
    ids = []

    while len(ids) < max_new_tokens:
        logits, _ = qwen3.prefill(tokens, weights, cfg)
        token = logits.argmax(dim=-1, keepdim=True)
        ids.append(token.item())
        if token.item() in eos_ids:
            break
        tokens = torch.cat([tokens, token])

    return ids


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


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        modal_common.MODEL_NAME,
        revision=MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        modal_common.MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def compare_prompt(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_ids = inputs["input_ids"][0]

    eos_ids = modal_common.eos_token_ids(model, tokenizer)

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
    if max_logit_diff > qwen3.MAX_LOGIT_DIFF:
        raise RuntimeError(
            f"prefill logits differ from hugging face by {max_logit_diff}, "
            f"which is above the tolerance of {qwen3.MAX_LOGIT_DIFF}"
        )

    return {
        "prompt": prompt,
        "ids": manual["ids"],
        "hf_completion": tokenizer.decode(expected["ids"], skip_special_tokens=True),
        "manual_completion": tokenizer.decode(manual["ids"], skip_special_tokens=True),
        "prefill_max_logit_diff": max_logit_diff,
        "kv_cache_shape_per_layer": manual["kv_cache_shape_per_layer"],
        "prefill_seconds": manual["prefill_seconds"],
        "decode_step_seconds": manual["decode_step_seconds"],
    }


@app.function(
    image=image,
    gpu=modal_common.GPU_TYPE,
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def manual_inference(prompt: str, max_new_tokens: int) -> dict[str, object]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    model, tokenizer = load_model()
    return compare_prompt(model, tokenizer, prompt, max_new_tokens)


@app.function(
    image=image,
    gpu=modal_common.GPU_TYPE,
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def regression_suite(
    cases: list[dict[str, object]], max_new_tokens: int
) -> list[dict[str, object]]:
    if not cases:
        raise ValueError("cases must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    model, tokenizer = load_model()
    results = []
    for case in cases:
        prompt = case["prompt"]
        token_ids = tokenizer(prompt)["input_ids"]
        if token_ids != case["token_ids"]:
            raise RuntimeError(
                f"prompt {prompt!r} tokenized to {token_ids}, "
                f"expected {case['token_ids']}"
            )
        result = compare_prompt(model, tokenizer, prompt, max_new_tokens)
        if result["ids"] != case["expected_token_ids"]:
            raise RuntimeError(
                f"prompt {prompt!r} generated {result['ids']}, "
                f"expected {case['expected_token_ids']}"
            )
        results.append(result)
    return results


@app.function(
    image=image,
    gpu=modal_common.GPU_TYPE,
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def cache_benchmark(
    prompt: str, max_new_tokens: int, prompt_repetitions: int
) -> dict[str, object]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if prompt_repetitions < 1:
        raise ValueError("prompt_repetitions must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    model, tokenizer = load_model()
    benchmark_prompt = " ".join([prompt] * prompt_repetitions)
    inputs = tokenizer(benchmark_prompt, return_tensors="pt").to("cuda")
    prompt_ids = inputs["input_ids"][0]
    weights = model.state_dict()

    eos_ids = modal_common.eos_token_ids(model, tokenizer)

    # warm up the kernels before either measurement
    with torch.inference_mode():
        qwen3.prefill(prompt_ids, weights, model.config)
    torch.cuda.synchronize()

    started = time.perf_counter()
    with torch.inference_mode():
        cached_ids = generate_cached(
            prompt_ids, weights, model.config, eos_ids, max_new_tokens
        )
    torch.cuda.synchronize()
    cached_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with torch.inference_mode():
        uncached_ids = generate_uncached(
            prompt_ids, weights, model.config, eos_ids, max_new_tokens
        )
    torch.cuda.synchronize()
    uncached_seconds = time.perf_counter() - started

    if cached_ids != uncached_ids:
        raise RuntimeError("cached and uncached generation produced different tokens")

    return {
        "prompt": prompt,
        "prompt_repetitions": prompt_repetitions,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(cached_ids),
        "cached_seconds": round(cached_seconds, 4),
        "uncached_seconds": round(uncached_seconds, 4),
        "speedup": round(uncached_seconds / cached_seconds, 2),
        "completion": tokenizer.decode(cached_ids, skip_special_tokens=True),
    }


def print_result(result):
    for key, value in result.items():
        print(f"{key}: {value}")


@app.local_entrypoint()
def main(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
    result = manual_inference.remote(prompt, max_new_tokens)
    print_result(result)


@app.local_entrypoint()
def run_regression() -> None:
    prompt_set = json.loads(PROMPT_SET_PATH.read_text(encoding="utf-8"))
    results = regression_suite.remote(
        prompt_set["cases"],
        prompt_set["max_new_tokens"],
    )

    print(f"regression prompts passed: {len(results)}")
    for result in results:
        print(
            f"{result['prompt']!r}: ids={result['ids']}, "
            f"max_logit_diff={result['prefill_max_logit_diff']}"
        )


@app.local_entrypoint()
def run_cache_benchmark() -> None:
    result = cache_benchmark.remote(
        DEFAULT_PROMPT,
        MAX_NEW_TOKENS,
        CACHE_BENCHMARK_PROMPT_REPETITIONS,
    )
    print_result(result)
