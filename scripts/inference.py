from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hida_baa.datasets import Sex
from hida_baa.predict import load_keras_model, predict_batch, resolve_single_image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bone age inference for one preprocessed 600x600 image.")
    parser.add_argument("input", help="Path to one preprocessed 600x600 image file.")
    parser.add_argument("--model", required=True, help="Path to a .keras model file.")
    parser.add_argument(
        "--sex",
        required=True,
        help="Sex metadata for the input image. Accepted: M/F/male/female/1/0.",
    )
    args = parser.parse_args()

    image_path = resolve_single_image_path(args.input)
    sex_value = Sex.normalize(args.sex)

    with contextlib.redirect_stdout(sys.stderr):
        model = load_keras_model(args.model)

    prediction = predict_batch(model, [image_path], [sex_value])[0]
    print(f"{float(prediction):.6f}")


if __name__ == "__main__":
    main()
