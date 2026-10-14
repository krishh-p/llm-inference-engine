from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

import modal_common

APP_NAME = "llm-inference-engine-fixtures"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 4
ROPE_ROTATION_ATOL = 1e-5
ROPE_ROTATION_RTOL = 0.0
ROPE_TABLE_ATOL = 1e-3
ROPE_TABLE_RTOL = 0.0
RMS_NORM_FLOAT32_ATOL = 1e-5
RMS_NORM_BFLOAT16_ATOL = 1e-2
# allowed error is atol + rtol * |expected|, one pair per stage of the block
BLOCK_TOLERANCES = {
    "norm": (1e-5, 1e-5),
    "projection": (1e-5, 1e-5),
    "rope": (1e-5, 1e-5),
    "scores": (1e-4, 1e-5),
    "probs": (1e-5, 1e-5),
    "attention_output": (1e-5, 1e-5),
    "mlp_output": (1e-5, 1e-5),
    "block_output": (1e-5, 1e-5),
}

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = modal_common.find_project_root(__file__)
MODEL_REVISION_PATH, MODEL_REVISION = modal_common.load_model_revision(PROJECT_ROOT)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "fixtures"
QWEN3_PATH = Path(__file__).with_name("qwen3.py")

app = modal.App(APP_NAME)

hf_cache = modal_common.hugging_face_cache()
image = modal_common.make_image(
    MODEL_REVISION_PATH,
    local_files=((QWEN3_PATH, "/root/qwen3.py"),),
)


with image.imports():
    import qwen3
    import torch


def on_cpu(tensor):
    return tensor.detach().cpu().contiguous()


def tensor_info(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def make_fixture(tensors):
    from safetensors.torch import save

    tensors = {name: on_cpu(tensor) for name, tensor in tensors.items()}
    return save(tensors), {
        name: tensor_info(tensor) for name, tensor in tensors.items()
    }


@app.function(
    image=image,
    gpu=modal_common.GPU_TYPE,
    volumes={modal_common.HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def build_fixtures() -> dict[str, object]:
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

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    prompt_ids = inputs["input_ids"][0]
    weights = model.state_dict()

    eos_ids = modal_common.eos_token_ids(model, tokenizer)

    trace = {}
    decode_trace = {}
    expected_ids = []

    with torch.inference_mode():
        logits, cache = qwen3.prefill(
            prompt_ids,
            weights,
            model.config,
            trace=trace,
            trace_layers=(0,),
        )

        while len(expected_ids) < MAX_NEW_TOKENS:
            token = logits.argmax(dim=-1, keepdim=True)
            expected_ids.append(token.item())
            if token.item() in eos_ids:
                break
            logits, cache = qwen3.decode_step(
                token,
                cache,
                weights,
                model.config,
                trace=decode_trace if len(expected_ids) == 1 else None,
                trace_layers=(0,),
            )

        hf_prefill_logits = model(**inputs).logits[0, -1, :]
        hf_output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    hf_ids = hf_output[0, len(prompt_ids) :].tolist()
    if expected_ids != hf_ids:
        raise RuntimeError(
            "fixture tokens do not match hugging face\n"
            f"expected: {hf_ids}\n"
            f"manual:   {expected_ids}"
        )

    max_logit_diff = (
        (trace["logits"].float() - hf_prefill_logits.float()).abs().max().item()
    )
    if max_logit_diff > qwen3.MAX_LOGIT_DIFF:
        raise RuntimeError(f"prefill logits differ by {max_logit_diff}")

    layer = "layer.0."
    layer_weights = "model.layers.0."
    rope_positions = torch.tensor(
        [0, 1, 128, 4096, 32767],
        device=prompt_ids.device,
    )
    rope_rows = len(rope_positions)

    # repeat one token so these positions don't depend on the prompt length
    q_input = trace[layer + "q_norm"][:, :1, :].expand(-1, rope_rows, -1).contiguous()
    k_input = trace[layer + "k_norm"][:, :1, :].expand(-1, rope_rows, -1).contiguous()
    rope_cos, rope_sin = qwen3.rope_tables(
        rope_positions,
        q_input.shape[-1],
        model.config.rope_theta,
    )
    rope_cos = rope_cos.to(q_input.dtype)
    rope_sin = rope_sin.to(q_input.dtype)
    q_rope = qwen3.apply_rope(q_input, rope_cos, rope_sin)
    k_rope = qwen3.apply_rope(k_input, rope_cos, rope_sin)

    q_norm_input = trace[layer + "q"].reshape(-1, model.config.head_dim)
    q_norm_weight = weights[layer_weights + "self_attn.q_norm.weight"]
    q_norm_output = trace[layer + "q_norm"].reshape(-1, model.config.head_dim)

    bf16_input = trace[layer + "input"].to(torch.bfloat16)
    bf16_weight = weights[layer_weights + "input_layernorm.weight"].to(torch.bfloat16)
    bf16_output = qwen3.rms_norm(
        bf16_input,
        bf16_weight,
        model.config.rms_norm_eps,
    )

    # 2560 is the hidden size used by Qwen3-4B. This keeps shape support from
    # accidentally depending on the 2048-wide model used to generate fixtures.
    wide_hidden_size = 2560
    wide_input = torch.linspace(
        -2.0,
        2.0,
        steps=2 * wide_hidden_size,
        device=prompt_ids.device,
        dtype=torch.float32,
    ).reshape(2, wide_hidden_size)
    wide_weight = torch.linspace(
        0.9,
        1.1,
        steps=wide_hidden_size,
        device=prompt_ids.device,
        dtype=torch.float32,
    )
    wide_output = qwen3.rms_norm(
        wide_input,
        wide_weight,
        model.config.rms_norm_eps,
    )

    cases = {
        "rms_norm.safetensors": {
            "input": trace[layer + "input"],
            "weight": weights[layer_weights + "input_layernorm.weight"],
            "eps": torch.tensor(model.config.rms_norm_eps, dtype=torch.float32),
            "output": trace[layer + "input_norm"],
            "q_input": q_norm_input,
            "q_weight": q_norm_weight,
            "q_output": q_norm_output,
            "bf16_input": bf16_input,
            "bf16_weight": bf16_weight,
            "bf16_output": bf16_output,
            "wide_input": wide_input,
            "wide_weight": wide_weight,
            "wide_output": wide_output,
        },
        "rope.safetensors": {
            "q_input": q_input,
            "k_input": k_input,
            "positions": rope_positions,
            "theta": torch.tensor(model.config.rope_theta, dtype=torch.float32),
            "cos": rope_cos,
            "sin": rope_sin,
            "q_output": q_rope,
            "k_output": k_rope,
        },
        "block_0.safetensors": {
            "input": trace[layer + "input"],
            "input_norm": trace[layer + "input_norm"],
            "q": trace[layer + "q"],
            "k": trace[layer + "k"],
            "v": trace[layer + "v"],
            "q_rope": trace[layer + "q_rope"],
            "k_rope": trace[layer + "k_rope"],
            "scores": trace[layer + "scores"],
            "attention_probs": trace[layer + "attention_probs"],
            "attention_output": trace[layer + "attention_output"],
            "mlp_input": trace[layer + "mlp_input"],
            "mlp_output": trace[layer + "mlp_output"],
            "output": trace[layer + "output"],
        },
        "decode_block_0.safetensors": {
            "token_id": torch.tensor([expected_ids[0]], dtype=torch.int64),
            "positions": decode_trace["positions"],
            "input": decode_trace[layer + "input"],
            "input_norm": decode_trace[layer + "input_norm"],
            "q": decode_trace[layer + "q"],
            "k": decode_trace[layer + "k"],
            "v": decode_trace[layer + "v"],
            "q_rope": decode_trace[layer + "q_rope"],
            "k_rope": decode_trace[layer + "k_rope"],
            "cache_k_in": decode_trace[layer + "cache_k_in"],
            "cache_v_in": decode_trace[layer + "cache_v_in"],
            "cache_k_out": decode_trace[layer + "cache_k_out"],
            "cache_v_out": decode_trace[layer + "cache_v_out"],
            "scores": decode_trace[layer + "scores"],
            "attention_probs": decode_trace[layer + "attention_probs"],
            "attention_output": decode_trace[layer + "attention_output"],
            "mlp_input": decode_trace[layer + "mlp_input"],
            "mlp_output": decode_trace[layer + "mlp_output"],
            "output": decode_trace[layer + "output"],
        },
        "generation.safetensors": {
            "final_norm": trace["final_norm"],
            "prefill_logits": trace["logits"],
        },
    }

    files = {}
    file_info = {}
    for filename, tensors in cases.items():
        files[filename], tensor_shapes = make_fixture(tensors)
        file_info[filename] = {"tensors": tensor_shapes}
    file_info["rope.safetensors"]["tolerances"] = {
        "rotation_atol": ROPE_ROTATION_ATOL,
        "rotation_rtol": ROPE_ROTATION_RTOL,
        "table_atol": ROPE_TABLE_ATOL,
        "table_rtol": ROPE_TABLE_RTOL,
    }
    file_info["rms_norm.safetensors"]["tolerances"] = {
        "float32_atol": RMS_NORM_FLOAT32_ATOL,
        "bfloat16_atol": RMS_NORM_BFLOAT16_ATOL,
    }
    for filename in ("block_0.safetensors", "decode_block_0.safetensors"):
        tolerances = {}
        for stage, (atol, rtol) in BLOCK_TOLERANCES.items():
            tolerances[f"{stage}_atol"] = atol
            tolerances[f"{stage}_rtol"] = rtol
        file_info[filename]["tolerances"] = tolerances

    manifest = {
        "format_version": 1,
        "model": {
            "id": modal_common.MODEL_NAME,
            "revision": MODEL_REVISION,
            "dtype": "float32",
        },
        "prompt": PROMPT,
        "token_ids": prompt_ids.tolist(),
        "positions": trace["positions"].tolist(),
        "expected_greedy_token_ids": expected_ids,
        "max_logit_diff_vs_hugging_face": max_logit_diff,
        "max_logit_diff_tolerance": qwen3.MAX_LOGIT_DIFF,
        "files": file_info,
    }

    return {"files": files, "manifest": manifest}


@app.local_entrypoint()
def main() -> None:
    result = build_fixtures.remote()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = result["manifest"]
    for filename, data in result["files"].items():
        path = OUTPUT_DIR / filename
        path.write_bytes(data)
        manifest["files"][filename]["bytes"] = len(data)
        manifest["files"][filename]["sha256"] = hashlib.sha256(data).hexdigest()

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(len(data) for data in result["files"].values())
    print(f"wrote {len(result['files'])} fixtures to {OUTPUT_DIR}")
    print(f"total_bytes: {total_bytes}")
    print(f"model_revision: {manifest['model']['revision']}")
