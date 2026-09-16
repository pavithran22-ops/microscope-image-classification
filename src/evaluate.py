"""
evaluate.py
===========
Evaluation module to compare Fully Connected Network (FCN) and
Convolutional Neural Network (CNN) on identical unseen microscope test images.

Calculates:
- Test Accuracy
- Test Loss
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Confusion Matrix
- Total and Trainable Parameters
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_loader import CLASS_NAMES, get_dataset


def load_test_data(results_dir: str = "results"):
    """Loads cached test data, or regenerates split deterministically if missing."""
    test_cache = os.path.join(results_dir, "test_data.npz")
    if os.path.exists(test_cache):
        data = np.load(test_cache)
        return data["X_test"], data["y_test"]
    else:
        splits = get_dataset(seed=42)
        return splits["X_test"], splits["y_test"]


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"):
    """
    Evaluates a trained Keras model on test data and returns detailed metrics.
    """
    # 1. Evaluate loss and accuracy via Keras
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # 2. Generate predictions
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # 3. Calculate classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # Model complexity
    total_params = int(model.count_params())

    metrics = {
        "model_name": model_name,
        "loss": float(test_loss),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "total_parameters": total_params,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }

    return metrics


def run_evaluation(
    model_dir: str = "models",
    results_dir: str = "results"
):
    """
    Evaluates both saved FCN and CNN models, prints a comparison table,
    and writes results to results/evaluation_metrics.json.
    """
    os.makedirs(results_dir, exist_ok=True)

    fcn_path = os.path.join(model_dir, "fcn_model.keras")
    cnn_path = os.path.join(model_dir, "cnn_model.keras")

    if not os.path.exists(fcn_path) or not os.path.exists(cnn_path):
        raise FileNotFoundError(
            f"Trained models not found! Expected '{fcn_path}' and '{cnn_path}'. "
            "Please run 'python src/train.py' first."
        )

    print("[INFO] Loading saved models...")
    fcn_model = tf.keras.models.load_model(fcn_path)
    cnn_model = tf.keras.models.load_model(cnn_path)

    print("[INFO] Loading test dataset...")
    X_test, y_test = load_test_data(results_dir)
    print(f"[INFO] Test dataset: {len(y_test)} images across classes: {CLASS_NAMES}")

    print("\n[INFO] Evaluating FCN (Fully Connected Network)...")
    fcn_metrics = evaluate_model(fcn_model, X_test, y_test, model_name="FCN (Dense/MLP)")

    print("[INFO] Evaluating CNN (Convolutional Neural Network)...")
    cnn_metrics = evaluate_model(cnn_model, X_test, y_test, model_name="CNN (Convolutional)")

    # Display comparison table
    print("\n" + "=" * 80)
    print("                FINAL BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Model Architecture':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Parameters':<12}")
    print("-" * 80)
    print(f"{'FCN (Fully Connected)':<22} | {fcn_metrics['accuracy']*100:>8.2f}% | {fcn_metrics['precision_macro']*100:>8.2f}% | {fcn_metrics['recall_macro']*100:>8.2f}% | {fcn_metrics['f1_macro']*100:>8.2f}% | {fcn_metrics['total_parameters']:>12,d}")
    print(f"{'CNN (Convolutional)':<22} | {cnn_metrics['accuracy']*100:>8.2f}% | {cnn_metrics['precision_macro']*100:>8.2f}% | {cnn_metrics['recall_macro']*100:>8.2f}% | {cnn_metrics['f1_macro']*100:>8.2f}% | {cnn_metrics['total_parameters']:>12,d}")
    print("=" * 80)

    # Save to JSON
    out_metrics = {
        "fcn": fcn_metrics,
        "cnn": cnn_metrics
    }
    metrics_file = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=4)
    print(f"[SUCCESS] Benchmark evaluation saved to: {metrics_file}")
    return out_metrics


if __name__ == "__main__":
    run_evaluation()
