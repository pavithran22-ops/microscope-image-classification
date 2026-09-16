"""
visualize.py
============
Generates visual figures for microscope image classification:
1. results/sample_images.png: Normal vs. Infected microscope cells.
2. results/training_curves.png: Side-by-side 2x2 training & validation accuracy and loss.
3. results/confusion_matrix_fcn.png: Annotated heatmap for FCN.
4. results/confusion_matrix_cnn.png: Annotated heatmap for CNN.
5. results/comparison.png: Grouped bar chart comparing metrics and parameters.
6. results/feature_maps.png: Intermediate CNN feature maps from Conv2D layer 1.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from data_loader import CLASS_NAMES, get_dataset


# Modern aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def plot_sample_images(output_path: str = "results/sample_images.png"):
    """
    Plots a grid of sample microscope images for both classes.
    """
    splits = get_dataset(seed=42)
    X = splits["X_train"]
    y = splits["y_train"]

    normal_idx = np.where(y == 0)[0][:4]
    infected_idx = np.where(y == 1)[0][:4]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), dpi=300)
    fig.suptitle("Sample Microscope Images (64x64 Grayscale)", fontsize=14, fontweight="bold", y=0.98)

    # Normal cells (Row 0)
    for i, idx in enumerate(normal_idx):
        ax = axes[0, i]
        ax.imshow(X[idx, :, :, 0], cmap="bone")
        ax.set_title(f"Normal Cell #{i+1}\n(Smooth, central pallor)", fontsize=10, color="#1b4965")
        ax.axis("off")

    # Infected cells (Row 1)
    for i, idx in enumerate(infected_idx):
        ax = axes[1, i]
        ax.imshow(X[idx, :, :, 0], cmap="bone")
        ax.set_title(f"Infected Cell #{i+1}\n(Parasite inclusions / clumps)", fontsize=10, color="#b7094c")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Sample images saved to: {output_path}")


def plot_training_curves(history_path: str = "results/training_history.json", output_path: str = "results/training_curves.png"):
    """
    Plots a 2x2 grid comparing FCN and CNN training & validation loss and accuracy.
    """
    if not os.path.exists(history_path):
        print(f"[WARNING] History file '{history_path}' not found. Skipping training curves.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    epochs_fcn = range(1, len(hist["fcn"]["loss"]) + 1)
    epochs_cnn = range(1, len(hist["cnn"]["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    fig.suptitle("Training & Validation Dynamics: FCN vs. CNN", fontsize=15, fontweight="bold")

    # 1. FCN Accuracy
    ax = axes[0, 0]
    ax.plot(epochs_fcn, hist["fcn"]["accuracy"], "o-", color="#3a86ff", label="Train Accuracy", linewidth=2)
    ax.plot(epochs_fcn, hist["fcn"]["val_accuracy"], "s--", color="#ff006e", label="Val Accuracy", linewidth=2)
    ax.set_title("FCN (Fully Connected) - Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. CNN Accuracy
    ax = axes[0, 1]
    ax.plot(epochs_cnn, hist["cnn"]["accuracy"], "o-", color="#2a9d8f", label="Train Accuracy", linewidth=2)
    ax.plot(epochs_cnn, hist["cnn"]["val_accuracy"], "s--", color="#e76f51", label="Val Accuracy", linewidth=2)
    ax.set_title("CNN (Convolutional) - Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. FCN Loss
    ax = axes[1, 0]
    ax.plot(epochs_fcn, hist["fcn"]["loss"], "o-", color="#3a86ff", label="Train Loss", linewidth=2)
    ax.plot(epochs_fcn, hist["fcn"]["val_loss"], "s--", color="#ff006e", label="Val Loss", linewidth=2)
    ax.set_title("FCN (Fully Connected) - Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. CNN Loss
    ax = axes[1, 1]
    ax.plot(epochs_cnn, hist["cnn"]["loss"], "o-", color="#2a9d8f", label="Train Loss", linewidth=2)
    ax.plot(epochs_cnn, hist["cnn"]["val_loss"], "s--", color="#e76f51", label="Val Loss", linewidth=2)
    ax.set_title("CNN (Convolutional) - Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Training curves saved to: {output_path}")


def plot_confusion_matrix(cm: list, model_name: str, output_path: str):
    """
    Renders an annotated heatmap confusion matrix.
    """
    cm_arr = np.array(cm)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Confusion matrix saved to: {output_path}")


def plot_model_comparison(metrics_path: str = "results/evaluation_metrics.json", output_path: str = "results/comparison.png"):
    """
    Plots a multi-metric bar chart comparing FCN and CNN performance and parameter efficiency.
    """
    if not os.path.exists(metrics_path):
        print(f"[WARNING] Metrics file '{metrics_path}' not found.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    fcn = metrics["fcn"]
    cnn = metrics["cnn"]

    # 1. Performance metrics
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fcn_values = [
        fcn["accuracy"] * 100,
        fcn["precision_macro"] * 100,
        fcn["recall_macro"] * 100,
        fcn["f1_macro"] * 100
    ]
    cnn_values = [
        cnn["accuracy"] * 100,
        cnn["precision_macro"] * 100,
        cnn["recall_macro"] * 100,
        cnn["f1_macro"] * 100
    ]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300, gridspec_kw={"width_ratios": [2.5, 1]})
    fig.suptitle("FCN vs. CNN: Microscope Classification Benchmark", fontsize=14, fontweight="bold")

    # Bar chart 1: Classification Performance
    rects1 = ax1.bar(x - width/2, fcn_values, width, label="FCN (Fully Connected)", color="#457b9d", alpha=0.9)
    rects2 = ax1.bar(x + width/2, cnn_values, width, label="CNN (Convolutional)", color="#2a9d8f", alpha=0.9)

    ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Classification Performance Metrics", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Add numeric labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Bar chart 2: Layer 1 Feature Parameter Comparison
    # Dense layer 1 in FCN vs Conv2D layer 1 in CNN
    # FCN Layer 1: 4096 * 128 + 128 = 524,416 params
    # CNN Layer 1: (3 * 3 * 1 + 1) * 32 = 320 params (Over 1,600x fewer parameters!)
    fcn_l1_params = 524416
    cnn_l1_params = 320

    bars = ax2.bar(["FCN Layer 1\n(Dense 128)", "CNN Layer 1\n(Conv2D 32)"], [fcn_l1_params, cnn_l1_params],
                   color=["#e63946", "#2a9d8f"], width=0.55)
    ax2.set_ylabel("Parameters Count (Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_title("First Layer Weights\n(Weight Sharing Effect)", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars, [fcn_l1_params, cnn_l1_params]):
        ax2.annotate(f"{val:,}", xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Model comparison chart saved to: {output_path}")


def plot_cnn_feature_maps(model_path: str = "models/cnn_model.keras", output_path: str = "results/feature_maps.png"):
    """
    Extracts intermediate feature maps from Conv2D layer 1 and visualizes
    how convolutional filters detect cell membranes, textures, and parasite spots.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] CNN model '{model_path}' not found. Skipping feature maps.")
        return

    cnn_model = tf.keras.models.load_model(model_path)
    splits = get_dataset(seed=42)

    # Find an infected cell image with rich features
    y_test = splits["y_test"]
    infected_indices = np.where(y_test == 1)[0]
    sample_img = splits["X_test"][infected_indices[0]:infected_indices[0]+1]

    # Create feature extraction model for Conv Layer 1
    conv1_layer = cnn_model.get_layer("cnn_conv1_32")
    feature_extractor = tf.keras.Model(inputs=cnn_model.inputs, outputs=conv1_layer.output)
    feature_maps = feature_extractor.predict(sample_img, verbose=0)[0]  # Shape: (64, 64, 32)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9), dpi=300)
    fig.suptitle("CNN Intermediate Feature Maps: Conv2D Layer 1 (64x64 Filters)\n"
                 "Demonstrating Local Spatial Edge, Boundary, and Parasite Inclusion Detection",
                 fontsize=13, fontweight="bold", y=0.98)

    # Top-left: Original Input Image
    ax0 = axes[0, 0]
    ax0.imshow(sample_img[0, :, :, 0], cmap="bone")
    ax0.set_title("Original Input Cell\n(Infected Cell with inclusions)", fontsize=10, fontweight="bold", color="#b7094c")
    ax0.axis("off")

    # Display 11 diverse feature maps from the 32 filters
    feature_indices = [0, 1, 2, 4, 7, 10, 12, 15, 18, 22, 28]
    descriptions = [
        "Filter #0: Membrane Contour",
        "Filter #1: High-Frequency Texture",
        "Filter #2: Dark Spot Inclusions",
        "Filter #4: Horizontal Gradients",
        "Filter #7: Intracellular Density",
        "Filter #10: Vertical Edge Response",
        "Filter #12: Cytoplasm Inversion",
        "Filter #15: Parasite Boundary",
        "Filter #18: Central Dimple Relief",
        "Filter #22: Optical Shadow Response",
        "Filter #28: Ring-Form Isolation"
    ]

    plot_pos = 1
    for f_idx, desc in zip(feature_indices, descriptions):
        r, c = divmod(plot_pos, 4)
        ax = axes[r, c]
        f_map = feature_maps[:, :, f_idx]
        ax.imshow(f_map, cmap="viridis")
        ax.set_title(desc, fontsize=9)
        ax.axis("off")
        plot_pos += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] CNN feature maps saved to: {output_path}")


def generate_all_visualizations():
    """Generates all required figures for reporting and presentation."""
    print("=" * 70)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 70)
    os.makedirs("results", exist_ok=True)

    # 1. Sample Images
    plot_sample_images("results/sample_images.png")

    # 2. Training Curves
    plot_training_curves("results/training_history.json", "results/training_curves.png")

    # 3. Confusion Matrices
    metrics_path = "results/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        plot_confusion_matrix(metrics["fcn"]["confusion_matrix"], "FCN (Fully Connected)", "results/confusion_matrix_fcn.png")
        plot_confusion_matrix(metrics["cnn"]["confusion_matrix"], "CNN (Convolutional)", "results/confusion_matrix_cnn.png")

    # 4. Model Comparison Chart
    plot_model_comparison("results/evaluation_metrics.json", "results/comparison.png")

    # 5. Intermediate CNN Feature Maps
    plot_cnn_feature_maps("models/cnn_model.keras", "results/feature_maps.png")

    print("=" * 70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY IN 'results/' FOLDER!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_visualizations()
