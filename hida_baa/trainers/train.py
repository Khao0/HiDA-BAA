from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

from hida_baa.datasets import data_generator_wrapper, debug_data, load_data, separate
from hida_baa.models import (
    SOURCE_MODEL_BACKBONE,
    SOURCE_MODEL_BACKBONE_LAYER,
    SOURCE_MODEL_INPUT_SIZE,
    load_source_model,
)
from hida_baa.trainers.utils import (
    evaluate,
    save_history,
    save_loss_graph,
    save_model_summary,
    save_result,
    scatter_plot,
)


REQUIRED_IMAGE_SIZE = SOURCE_MODEL_INPUT_SIZE


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open() as file:
        return yaml.safe_load(file)


def _validate_config(config: dict[str, Any]) -> None:
    image_size = list(config["data"].get("image_size", REQUIRED_IMAGE_SIZE))
    if image_size != REQUIRED_IMAGE_SIZE:
        raise ValueError(
            f"HiDA-SourceModel was trained with {SOURCE_MODEL_BACKBONE} at 600x600 resolution; "
            f"this repository expects image_size: {REQUIRED_IMAGE_SIZE}, got {image_size}."
        )


def _prepare_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    training_config = config["training"]
    run_name = training_config.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("target_adaptation_%Y%m%d-%H%M%S")

    checkpoint_dir = Path(training_config.get("checkpoint_dir", "checkpoints/AdaptedModel")) / run_name

    dirs = {
        "save": checkpoint_dir,
        "loss": checkpoint_dir / "loss_graphs",
        "history": checkpoint_dir / "history",
        "predictions": checkpoint_dir / "predictions",
        "weights": checkpoint_dir / "weights",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs, run_name


def _make_optimizer(learning_rate: float, weight_decay: float | None = None) -> tf.keras.optimizers.Optimizer:
    if weight_decay is None:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    try:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    except TypeError:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _set_callbacks(save_path: str | Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            save_weights_only=False,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-8,
        ),
    ]


def _find_backbone(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    try:
        return model.get_layer(layer_name)
    except ValueError as exc:
        layer_names = [layer.name for layer in model.layers]
        raise ValueError(
            f"Backbone layer {layer_name!r} not found in HiDA-SourceModel. "
            f"Available top-level layers: {layer_names}"
        ) from exc


def _compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    loss: str,
    metrics: list[str],
    weight_decay: float | None = None,
) -> None:
    model.compile(
        optimizer=_make_optimizer(learning_rate, weight_decay),
        loss=loss,
        metrics=metrics,
    )


def _save_resolved_config(config: dict[str, Any], save_dir: Path) -> None:
    with (save_dir / "config.resolved.json").open("w") as file:
        json.dump(config, file, indent=2)


def run_training(config_path: str | Path) -> dict[int, list[float]]:
    config = load_config(config_path)
    _validate_config(config)

    tf.keras.utils.set_random_seed(int(config.get("seed", 42)))
    print("TensorFlow Version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))

    data_config = config["data"]
    training_config = config["training"]
    fine_tune_config = config.get("fine_tuning", {})

    batch_size = int(data_config.get("batch_size", 32))
    n_folds = int(data_config.get("k_fold", 10))
    init_epochs = int(training_config.get("warm_up_epochs", 10))
    epochs = int(training_config.get("epochs", 100))
    patience = int(training_config.get("patience", 10))
    warm_up_lr = float(training_config.get("warm_up_learning_rate", 1e-4))
    lr = float(training_config.get("learning_rate", 1e-5))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    warm_up_loss = training_config.get("warm_up_loss", training_config.get("loss", "mean_absolute_error"))
    fine_tune_loss = training_config.get("fine_tune_loss", "mean_absolute_error")
    metrics = training_config.get("metrics", ["mae"])

    output_dirs, run_name = _prepare_output_dirs(config)
    _save_resolved_config(config, output_dirs["save"])

    train_data = load_data(config, train=True)
    test_data = load_data(config, train=False)

    test_dataset = data_generator_wrapper(
        test_data,
        config,
        image_root=data_config["test_path"],
        shuffle=False,
    )
    test_steps = int(np.ceil(len(test_data) / batch_size))

    cv_results: dict[int, list[float]] = {}
    backbone_layer_name = SOURCE_MODEL_BACKBONE_LAYER
    unfreeze_from = int(fine_tune_config.get("unfreeze_from_layer", 713))

    for fold_id in range(n_folds):
        print(f"\n========== Fold {fold_id + 1}/{n_folds} ==========")

        train_set, train_step, val_set, val_step = separate(train_data, fold_id, batch_size)
        train_dataset = data_generator_wrapper(
            train_set,
            config,
            image_root=data_config["train_path"],
            shuffle=True,
        )
        val_dataset = data_generator_wrapper(
            val_set,
            config,
            image_root=data_config["train_path"],
            shuffle=False,
        )

        debug_data(train_set, config, image_root=data_config["train_path"])

        save_model_i_path = output_dirs["weights"] / f"best_model_fold{fold_id}.keras"
        callbacks = _set_callbacks(save_model_i_path, patience)

        model = load_source_model()
        model.trainable = True
        backbone = _find_backbone(model, backbone_layer_name)
        backbone.trainable = False

        _compile_model(model, warm_up_lr, warm_up_loss, metrics)
        save_model_summary(model, output_dirs["save"] / f"model_summary_warmup.txt", fold_id)

        if fold_id == 0:
            try:
                tf.keras.utils.plot_model(
                    model,
                    output_dirs["save"] / "structure.png",
                    show_shapes=False,
                )
            except Exception as exc:  # pragma: no cover - optional graphviz dependency
                print(f"[WARN] Could not save model structure image: {exc}")

        print("Training regression head warm-up...")
        model.fit(
            train_dataset,
            steps_per_epoch=train_step,
            validation_data=val_dataset,
            validation_steps=val_step,
            epochs=init_epochs,
            callbacks=callbacks,
        )

        print("Fine-tuning HiDA-TargetModel...")
        model = tf.keras.models.load_model(save_model_i_path)
        model.trainable = True
        backbone = _find_backbone(model, backbone_layer_name)
        for idx, layer in enumerate(backbone.layers):
            layer.trainable = idx > unfreeze_from

        _compile_model(model, lr, fine_tune_loss, metrics, weight_decay=weight_decay)
    
        save_model_summary(model, output_dirs["save"] / f"model_summary_finetune.txt", fold_id)

        history = model.fit(
            train_dataset,
            steps_per_epoch=train_step,
            validation_data=val_dataset,
            validation_steps=val_step,
            epochs=epochs,
            callbacks=callbacks,
        )

        save_loss_graph(history, output_dirs["loss"], fold_id)
        save_history(history, output_dirs["history"], fold_id)

        mae, rmse, r2, prediction_df = evaluate(test_dataset, model, test_steps)
        cv_results[fold_id] = [mae, rmse, r2]
        scatter_plot(prediction_df, output_dirs["predictions"], f"predictions_fold{fold_id}")

    save_result(cv_results, output_dirs["save"], "cv_results")
    checkpoint_path = Path(training_config.get("checkpoint_dir", "checkpoints/AdaptedModel")) / run_name


    print("\n" + "="*50)
    print("✅ Training Completed Successfully")
    print("="*50)

    print(f"📁 Checkpoint saved at:\n   {checkpoint_path}")

    print("\n📊 Cross-validation Results:")
    print("-"*50)

    # Pretty print dict / list nicely
    if isinstance(cv_results, (dict, list)):
        print(json.dumps(cv_results, indent=4))
    else:
        print(cv_results)

    print("="*50 + "\n")
    return cv_results
