from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(dataset: tf.data.Dataset, model: tf.keras.Model, steps: int) -> tuple[float, float, float, pd.DataFrame]:
    print("Evaluating on test set...")
    model.trainable = False

    y_true = []
    y_pred = []

    for i, (x_batch, y_batch) in enumerate(dataset):
        if i >= steps:
            break

        preds = model.predict(x_batch, verbose=0).flatten()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

    result_df = pd.DataFrame(
        {
            "GT": np.array(y_true, dtype=np.float32),
            "PD": np.array(y_pred, dtype=np.float32),
        }
    )

    mae = mean_absolute_error(result_df["GT"], result_df["PD"])
    rmse = np.sqrt(mean_squared_error(result_df["GT"], result_df["PD"]))
    r2 = r2_score(result_df["GT"], result_df["PD"])

    return float(mae), float(rmse), float(r2), result_df


def save_loss_graph(history: tf.keras.callbacks.History, save_dir: str | Path, fold_id: int) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history_data = history.history
    plt.figure()
    plt.plot(history_data.get("loss", []), label="train_loss")
    plt.plot(history_data.get("val_loss", []), label="val_loss")
    plt.title(f"Loss Fold {fold_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"loss_fold{fold_id}.png")
    plt.close()


def scatter_plot(df: pd.DataFrame, save_dir: str | Path, output_name: str) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sns.scatterplot(x="PD", y="GT", data=df)
    max_val = max(df["PD"].max(), df["GT"].max())
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title("Ground Truth vs Prediction Scatter Plot")
    plt.xlabel("Predicted Bone Age (months)")
    plt.ylabel("Ground Truth Bone Age (months)")
    plt.tight_layout()
    plt.savefig(save_dir / f"{output_name}.png")
    plt.close()
    df.to_csv(save_dir / f"{output_name}.csv", index=False)


def save_history(history: tf.keras.callbacks.History, save_dir: str | Path, fold_id: int) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / f"history_fold{fold_id}.pkl").open("wb") as file:
        pickle.dump(history.history, file)


def save_result(results: dict[int, list[float]], save_dir: str | Path, output_name: str) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(results, orient="index", columns=["mae", "rmse", "r2"])
    df.index.name = "fold"
    mean_row = df.mean(numeric_only=True)
    std_row = df.std(numeric_only=True)
    df.loc["mean"] = mean_row
    df.loc["std"] = std_row
    df.to_csv(save_dir / f"{output_name}.csv")

    json_ready: dict[str, Any] = {
        str(key): [float(metric) for metric in value] for key, value in results.items()
    }
    json_ready["mean"] = {key: float(value) for key, value in df.loc["mean"].to_dict().items()}
    json_ready["std"] = {key: float(value) for key, value in df.loc["std"].to_dict().items()}
    with (save_dir / f"{output_name}.json").open("w") as file:
        json.dump(json_ready, file, indent=2)


def save_model_summary(model: tf.keras.Model, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def append_line(line: str, *args: Any, **kwargs: Any) -> None:
        lines.append(line)

    model.summary(print_fn=append_line)
    save_path.write_text("\n".join(lines))
