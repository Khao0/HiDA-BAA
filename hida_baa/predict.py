from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from hida_baa.datasets import Sex
from hida_baa.models import SOURCE_MODEL_INPUT_SIZE


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_keras_model(model_path: str | Path) -> tf.keras.Model:
    model_path = Path(model_path)
    if model_path.suffix != ".keras":
        raise ValueError(f"Model file must use .keras format: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not model_path.is_file():
        raise ValueError(f"Model path must be a file: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def resolve_single_image_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Inference accepts one image file at a time: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {input_path.suffix}")
    return input_path


def resolve_image_path(image_root: str | Path, image_name: str, image_extension: str = ".png") -> Path:
    image_path = Path(str(image_name))
    if not image_path.suffix and image_extension:
        image_path = image_path.with_suffix(image_extension)
    if not image_path.is_absolute():
        image_path = Path(image_root) / image_path
    return image_path


def read_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    image_size = tuple(SOURCE_MODEL_INPUT_SIZE)
    if img.shape[:2] != image_size:
        raise ValueError(
            f"Input image must be {image_size[0]}x{image_size[1]} pixels, "
            f"got {img.shape[0]}x{img.shape[1]} for {image_path}"
        )

    return img.astype(np.float32)


def predict_batch(
    model: tf.keras.Model,
    image_paths: list[Path],
    sex_values: list[int],
) -> np.ndarray:
    images = np.array([read_image(path) for path in image_paths], dtype=np.float32)
    sexes = np.array(sex_values, dtype=np.float32)
    return model.predict((images, sexes), verbose=0).reshape(-1)


def predict_dataframe(
    model: tf.keras.Model,
    df: pd.DataFrame,
    image_root: str | Path,
    image_column: str = "image_name",
    sex_column: str = "sex",
    image_extension: str = ".png",
) -> pd.DataFrame:
    result = df.copy()
    image_paths = [
        resolve_image_path(image_root, image_name, image_extension)
        for image_name in result[image_column].tolist()
    ]
    sex_values = [Sex.normalize(value) for value in result[sex_column].tolist()]
    result["predicted_bone_age"] = predict_batch(model, image_paths, sex_values)
    return result
