from __future__ import annotations

import modal

APP_NAME = "llm-inference-engine-dev"
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
HF_CACHE_PATH = "/cache/huggingface"

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
)


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def smoke_test() -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")

    gpu_name = torch.cuda.get_device_name()
    cuda_version = str(torch.version.cuda)
    torch_version = str(torch.__version__)

    print(f"GPU: {gpu_name}")
    print(f"CUDA: {cuda_version}")
    print(f"PyTorch: {torch_version}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"the model was loaded on {model_device}, not CUDA")

    inputs = tokenizer(
        "The capital of France is",
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    completion = tokenizer.decode(
        outputs[0, prompt_length:],
        skip_special_tokens=True,
    )
    print(f"completion: {completion}")

    return {
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "cuda": cuda_version,
        "pytorch": torch_version,
        "completion": completion,
    }


@app.local_entrypoint()
def main() -> None:
    result = smoke_test.remote()

    print("smoke test passed")
    for key, value in result.items():
        print(f"{key}: {value}")
