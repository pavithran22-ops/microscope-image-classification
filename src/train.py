"""
train.py
========
Training pipeline for comparing Fully Connected Network (FCN) and
Convolutional Neural Network (CNN) on microscope image classification.

Key features:
1. Identical training conditions (epochs, batch size, learning rate, dataset splits).
2. EarlyStopping to prevent overfitting.
3. Saves models to models/fcn_model.keras and models/cnn_model.keras.
4. Serializes training history to results/training_history.json for visualization.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

from data_loader import get_dataset, create_tf_datasets
from models import build_fcn_model, build_cnn_model, compile_model


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and TensorFlow for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_both_models(
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    output_model_dir: str = "models",
    output_results_dir: str = "results"
):
    """
    Orchestrates end-to-end training of FCN and CNN under identical conditions.
    """
    set_seed(seed)

    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_results_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: PREPARING DATASET & SPLITS (IDENTICAL FOR BOTH MODELS)")
    print("=" * 70)
    splits = get_dataset(seed=seed)
    train_ds, val_ds, _ = create_tf_datasets(splits, batch_size=batch_size)

    # Save test dataset split for evaluate.py
    test_data_path = os.path.join(output_results_dir, "test_data.npz")
    np.savez_compressed(
        test_data_path,
        X_test=splits["X_test"],
        y_test=splits["y_test"]
    )
    print(f"[INFO] Test data cached for evaluation at: {test_data_path}")

    # Shared EarlyStopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # =========================================================================
    # MODEL 1: FULLY CONNECTED NETWORK (FCN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING FULLY CONNECTED NETWORK (FCN / MLP)")
    print("Notice: 2D image is unrolled into 1D (4,096 scalar inputs).")
    print("=" * 70)
    fcn_model = build_fcn_model()
    compile_model(fcn_model, learning_rate=learning_rate)
    fcn_model.summary()

    fcn_history = fcn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    fcn_save_path = os.path.join(output_model_dir, "fcn_model.keras")
    fcn_model.save(fcn_save_path)
    print(f"[SUCCESS] FCN model saved to: {fcn_save_path}")

    # =========================================================================
    # MODEL 2: CONVOLUTIONAL NEURAL NETWORK (CNN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("Notice: 2D spatial context is preserved with sliding local kernels.")
    print("=" * 70)
    cnn_model = build_cnn_model()
    compile_model(cnn_model, learning_rate=learning_rate)
    cnn_model.summary()

    cnn_history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    cnn_save_path = os.path.join(output_model_dir, "cnn_model.keras")
    cnn_model.save(cnn_save_path)
    print(f"[SUCCESS] CNN model saved to: {cnn_save_path}")

    # =========================================================================
    # SAVE TRAINING HISTORIES TO JSON
    # =========================================================================
    history_record = {
        "epochs_requested": epochs,
        "fcn": {k: [float(v) for v in vals] for k, vals in fcn_history.history.items()},
        "cnn": {k: [float(v) for v in vals] for k, vals in cnn_history.history.items()}
    }

    history_file = os.path.join(output_results_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_record, f, indent=4)
    print(f"[SUCCESS] Training metrics saved to: {history_file}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! PROCEED TO RUN: python src/evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    train_both_models(epochs=18, batch_size=32, learning_rate=0.001, seed=42)
