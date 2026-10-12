from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
GPU_TYPE = "L4"
HF_CACHE_PATH = "/cache/huggingface"


def find_project_root(source_file):
    source_dir = Path(source_file).resolve().parent
    for path in (source_dir, *source_dir.parents):
        if (path / "model_revision.txt").is_file():
            return path
    raise FileNotFoundError(
        "model_revision.txt is missing; include it in the repository"
    )


def load_model_revision(project_root):
    path = project_root / "model_revision.txt"
    return path, path.read_text(encoding="utf-8").strip()


def make_image(model_revision_path, local_files=()):
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
        .add_local_file(Path(__file__).resolve(), "/root/modal_common.py")
        .add_local_file(model_revision_path, "/root/model_revision.txt")
    )

    for local_path, remote_path in local_files:
        image = image.add_local_file(local_path, remote_path)

    return image


def hugging_face_cache():
    return modal.Volume.from_name(
        "llm-inference-engine-hf-cache",
        create_if_missing=True,
    )


def eos_token_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return [ids] if isinstance(ids, int) else ids
