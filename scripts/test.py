from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hida_baa.predict import load_keras_model, predict_dataframe


def _resolve_test_csv(test_folder: Path, csv_path: str | None) -> Path:
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Test CSV not found: {path}")

    candidates = [
        test_folder / "test.csv",
        test_folder.parent / "test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find test.csv in {test_folder} or {test_folder.parent}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a test folder with test.csv metadata.")
    parser.add_argument("test_folder", help="Folder containing preprocessed 600x600 test images.")
    parser.add_argument("--model", required=True, help="Path to a .keras model file.")
    parser.add_argument("--csv", default=None, help="Optional explicit path to test.csv.")
    parser.add_argument("--image-column", default="image_name", help="Image-name column in test.csv.")
    parser.add_argument("--label-column", default="bone_age", help="Bone-age label column in test.csv.")
    parser.add_argument("--sex-column", default="sex", help="Sex column in test.csv.")
    parser.add_argument("--image-extension", default=".png", help="Extension to use when image names omit suffix.")
    args = parser.parse_args()

    test_folder = Path(args.test_folder)
    if not test_folder.exists() or not test_folder.is_dir():
        raise NotADirectoryError(f"Test folder not found: {test_folder}")

    test_csv = _resolve_test_csv(test_folder, args.csv)
    df = pd.read_csv(test_csv)

    required_columns = {args.image_column, args.label_column, args.sex_column}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{test_csv} is missing columns: {missing_text}")

    with contextlib.redirect_stdout(sys.stderr):
        model = load_keras_model(args.model)
        result_df = predict_dataframe(
            model,
            df,
            image_root=test_folder,
            image_column=args.image_column,
            sex_column=args.sex_column,
            image_extension=args.image_extension,
        )

    y_true = result_df[args.label_column].astype(float).to_numpy()
    y_pred = result_df["predicted_bone_age"].astype(float).to_numpy()

    output = {
        "n": int(len(result_df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
