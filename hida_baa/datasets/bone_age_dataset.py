from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


class Sex(int, Enum):
    male = 1
    female = 0

    @classmethod
    def normalize(cls, value: str | int | float) -> int:
        value = str(value).strip().lower()
        if value in {"m", "male", "1"}:
            return int(cls.male)
        if value in {"f", "female", "0"}:
            return int(cls.female)
        raise ValueError(f"Invalid sex value: {value!r}")


def load_data(config: dict[str, Any], train: bool = True) -> pd.DataFrame:
    data_config = config["data"]
    annotations_path = Path(
        data_config["train_annotations"] if train else data_config["test_annotations"]
    )

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

    df = pd.read_csv(annotations_path)
    image_column = data_config.get("image_column", "image_name")
    label_column = data_config.get("label_column", "bone_age")
    sex_column = data_config.get("sex_column", "sex")

    required_columns = {image_column, label_column, sex_column}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{annotations_path} is missing columns: {missing_text}")

    df = df.copy()
    df[sex_column] = df[sex_column].map(Sex.normalize)
    df[label_column] = df[label_column].astype(float)

    if train:
        df = assign_folds(df, config)

    return df


def assign_folds(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    label_column = data_config.get("label_column", "bone_age")
    sex_column = data_config.get("sex_column", "sex")
    n_splits = int(data_config.get("k_fold", 10))
    seed = int(config.get("seed", 42))

    df = df.copy()
    age_year = (df[label_column].astype(float) // 12).astype(int)
    stratify_key = age_year.astype(str) + "_" + df[sex_column].astype(str)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(splitter.split(df, stratify_key)):
        df.loc[df.index[val_idx], "fold"] = fold_id

    return df


def _resolve_image_path(image_root: str | Path, image_name: str, extension: str) -> Path:
    image_path = Path(str(image_name))
    if not image_path.suffix and extension:
        image_path = image_path.with_suffix(extension)
    if not image_path.is_absolute():
        image_path = Path(image_root) / image_path
    return image_path


def _read_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    if img.shape[:2] != image_size:
        raise ValueError(
            f"Input image must be {image_size[0]}x{image_size[1]} pixels, "
            f"got {img.shape[0]}x{img.shape[1]} for {image_path}"
        )

    return img.astype(np.float32)


def data_generator(
    data: tuple[tuple[str, int, float], ...],
    image_root: str | Path,
    image_size: tuple[int, int],
    image_extension: str,
    batch_size: int,
    shuffle: bool,
):
    while True:
        indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(data), batch_size):
            batch_idx = indices[start : start + batch_size]

            batch_x_img = []
            batch_x_sex = []
            batch_y = []

            for i in batch_idx:
                image_name, sex, bone_age = data[i]
                image_path = _resolve_image_path(image_root, image_name, image_extension)
                img = _read_image(image_path, image_size)

                batch_x_img.append(img)
                batch_x_sex.append(sex)
                batch_y.append(bone_age)

            yield (
                (
                    np.array(batch_x_img, dtype=np.float32),
                    np.array(batch_x_sex, dtype=np.float32),
                ),
                np.array(batch_y, dtype=np.float32),
            )


def data_generator_wrapper(
    df: pd.DataFrame,
    config: dict[str, Any],
    image_root: str | Path,
    shuffle: bool = True,
) -> tf.data.Dataset:
    data_config = config["data"]
    image_column = data_config.get("image_column", "image_name")
    label_column = data_config.get("label_column", "bone_age")
    sex_column = data_config.get("sex_column", "sex")
    image_size = tuple(data_config.get("image_size", [600, 600]))
    image_extension = data_config.get("image_extension", ".png")
    batch_size = int(data_config.get("batch_size", 32))

    data = tuple(df[[image_column, sex_column, label_column]].itertuples(index=False, name=None))
    generator = data_generator(data, image_root, image_size, image_extension, batch_size, shuffle)

    output_signature = (
        (
            tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=output_signature,
    )


def separate(
    train_data: pd.DataFrame,
    fold_id: int,
    batch_size: int,
) -> tuple[pd.DataFrame, int, pd.DataFrame, int]:
    train_set = train_data[train_data["fold"] != fold_id].reset_index(drop=True)
    val_set = train_data[train_data["fold"] == fold_id].reset_index(drop=True)

    train_step = int(np.ceil(len(train_set) / batch_size))
    val_step = int(np.ceil(len(val_set) / batch_size))

    return train_set, train_step, val_set, val_step


def debug_data(df: pd.DataFrame, config: dict[str, Any], image_root: str | Path) -> None:
    dataset = data_generator_wrapper(df, config, image_root=image_root, shuffle=False)
    for inputs, labels in dataset.take(1):
        print(f"Shape of input images: {inputs[0].shape}")
        print(f"Shape of input gender: {inputs[1].shape}")
        print(f"Shape of labels: {labels.shape}")
