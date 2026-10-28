"""Train the CNN-LSTM model from collected .npy landmark sequences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from sign_language.model import build_model


def load_dataset(data_dir: Path, sequence_length: int):
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    class_names: list[str] = []
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    for class_dir in class_dirs:
        files = sorted(class_dir.glob("*.npy"))
        class_sequences: list[np.ndarray] = []
        for file in files:
            sequence = np.load(file, allow_pickle=False)
            if sequence.shape != (sequence_length, 42, 3):
                raise ValueError(
                    f"{file} has shape {sequence.shape}; expected "
                    f"({sequence_length}, 42, 3). Recollect it using the same "
                    "sequence length as training."
                )
            if not np.isfinite(sequence).all():
                raise ValueError(f"{file} contains non-finite landmark values.")
            if not np.any(sequence):
                print(f"Skipping {file}: no hand landmarks were detected.")
                continue
            class_sequences.append(sequence.astype(np.float32, copy=False))
        if not class_sequences:
            print(f"Skipping label '{class_dir.name}': no usable hand samples.")
            continue
        class_index = len(class_names)
        class_names.append(class_dir.name)
        sequences.extend(class_sequences)
        targets.extend([class_index] * len(class_sequences))

    if len(class_names) < 2:
        raise ValueError(
            f"Expected at least two labels with hand landmarks in {data_dir}."
        )

    return np.stack(sequences), np.asarray(targets, dtype=np.int64), class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sequences"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data}")
    if args.epochs < 1 or args.batch_size < 1 or args.sequence_length < 2:
        parser.error("epochs and batch-size must be positive; sequence-length must be >= 2")

    try:
        x, y, class_names = load_dataset(args.data, args.sequence_length)
    except ValueError as exc:
        parser.error(str(exc))

    class_counts = np.bincount(y, minlength=len(class_names))
    has_validation_set = bool(np.all(class_counts >= 2))
    if has_validation_set:
        validation_count = max(len(class_names), math.ceil(len(y) * 0.2))
        if validation_count > len(y) - len(class_names):
            parser.error("Not enough samples for a stratified train/validation split.")
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=validation_count,
            random_state=42,
            stratify=y,
        )
    else:
        print(
            "Warning: at least one label has only one sample. Training without "
            "independent validation; the saved model is a demo, not a measure "
            "of generalization."
        )
        x_train, y_train = x, y
        x_val, y_val = None, None

    model = build_model(args.sequence_length, len(class_names))
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "sign_cnn_lstm.keras"
    monitor = "val_accuracy" if has_validation_set else "loss"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path, monitor=monitor, save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if has_validation_set else "loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if has_validation_set else "loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        ),
    ]
    fit_args = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "callbacks": callbacks,
    }
    if has_validation_set:
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val), **fit_args
        )
    else:
        history = model.fit(x_train, y_train, **fit_args)
    model = keras.models.load_model(model_path)
    if has_validation_set:
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
        training_loss = None
        training_accuracy = None
    else:
        training_loss, training_accuracy = model.evaluate(x_train, y_train, verbose=0)
        loss = None
        accuracy = None
    with (args.output / "labels.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    with (args.output / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_accuracy": float(accuracy) if accuracy is not None else None,
                "validation_loss": float(loss) if loss is not None else None,
                "training_accuracy": (
                    float(training_accuracy)
                    if training_accuracy is not None
                    else None
                ),
                "training_loss": (
                    float(training_loss) if training_loss is not None else None
                ),
                "evaluation": (
                    "held_out_validation"
                    if has_validation_set
                    else "training_only_no_independent_validation"
                ),
                "samples": int(len(y)),
                "classes": class_names,
                "samples_per_class": {
                    class_name: int(class_counts[index])
                    for index, class_name in enumerate(class_names)
                },
                "epochs_completed": len(history.history["loss"]),
            },
            file,
            indent=2,
        )
    print(f"Saved model to {model_path}")
    if has_validation_set:
        print(f"Validation accuracy: {accuracy:.3f}")
    else:
        print(
            f"Training accuracy (not validation accuracy): {training_accuracy:.3f}. "
            "Collect more samples per letter before relying on predictions."
        )


if __name__ == "__main__":
    main()
