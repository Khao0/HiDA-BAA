from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from huggingface_hub import hf_hub_download


SOURCE_MODEL_NAME = "HiDA-SourceModel"
SOURCE_MODEL_FILENAME = "HiDA-SourceModel.keras"
SOURCE_MODEL_REPO_ID = "Kwankhao/HiDA-SourceModel"
SOURCE_MODEL_PATH = Path("checkpoints/SourceModel") / SOURCE_MODEL_FILENAME
SOURCE_MODEL_BACKBONE = "EfficientNetB7"
SOURCE_MODEL_BACKBONE_LAYER = "efficientnetb7"
SOURCE_MODEL_INPUT_SIZE = [600, 600]


def resolve_source_model(
    source_model_path: str | Path = SOURCE_MODEL_PATH,
    repo_id: str = SOURCE_MODEL_REPO_ID,
    filename: str = SOURCE_MODEL_FILENAME,
) -> str:
    source_model_path = Path(source_model_path)
    if source_model_path.exists():
        print(f"[INFO] Using existing HiDA-SourceModel: {source_model_path}")
        return str(source_model_path)

    print("[INFO] HiDA-SourceModel not found locally. Downloading from Hugging Face...")
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"[INFO] Downloaded HiDA-SourceModel to: {downloaded_path}")
    return downloaded_path


def load_source_model(
    source_model_path: str | Path = SOURCE_MODEL_PATH,
    repo_id: str = SOURCE_MODEL_REPO_ID,
    filename: str = SOURCE_MODEL_FILENAME,
) -> tf.keras.Model:
    source_model_path = resolve_source_model(source_model_path, repo_id=repo_id, filename=filename)
    model = tf.keras.models.load_model(source_model_path, compile=False)
    return model
