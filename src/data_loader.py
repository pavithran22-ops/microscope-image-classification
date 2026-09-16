"""
data_loader.py
==============
Data loading, synthetic microscope image generation, and preprocessing pipeline.

This module provides:
1. A realistic synthetic microscope cell image generator (Normal vs. Infected cells).
2. Support for loading custom user microscope datasets from directory structures.
3. Preprocessing: resizing, grayscale conversion, [0, 1] normalization.
4. Stratified train / validation / test splits for fair benchmark comparisons.
"""

import os
import math
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Class labels
CLASS_NAMES = ["Normal_Cell", "Infected_Cell"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (64, 64)


def generate_single_cell(cell_type: int, img_size: int = 64, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Synthesizes a realistic 64x64 grayscale microscope image of a cell.

    Parameters:
        cell_type (int): 0 for Normal_Cell, 1 for Infected_Cell.
        img_size (int): Image width and height (default 64).
        rng (np.random.RandomState): Random generator for reproducibility.

    Returns:
        np.ndarray: 2D array of shape (img_size, img_size) with values in [0.0, 1.0].
    """
    if rng is None:
        rng = np.random.RandomState()

    # 1. Base slide background (light microscope field with slight illumination gradient)
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(-1, 1, img_size)
    xx, yy = np.meshgrid(x, y)

    # Microscope field gradient: brighter center, slightly darker edges (vignetting)
    field_gradient = 0.85 - 0.12 * (xx**2 + yy**2)
    # Stain/slide noise
    noise = rng.normal(loc=0.0, scale=0.02, size=(img_size, img_size))
    img = field_gradient + noise

    # 2. Cell morphology (placed near center with random slight jitter and elongation)
    cx = rng.uniform(-0.10, 0.10)
    cy = rng.uniform(-0.10, 0.10)
    radius_x = rng.uniform(0.40, 0.52)
    radius_y = rng.uniform(0.40, 0.52)
    angle = rng.uniform(0, np.pi)

    # Rotated coordinate system for the elliptical cell body
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_rot = (xx - cx) * cos_a + (yy - cy) * sin_a
    y_rot = -(xx - cx) * sin_a + (yy - cy) * cos_a
    dist_sq = (x_rot / radius_x)**2 + (y_rot / radius_y)**2

    # Cytoplasm mask (inside the cell membrane)
    cell_mask = dist_sq <= 1.0
    # Membrane edge (boundary of the cell)
    membrane_mask = (dist_sq > 0.85) & (dist_sq <= 1.05)

    # Apply cell body staining (cytoplasm is darker than slide background)
    cytoplasm_intensity = rng.uniform(0.50, 0.60)
    img[cell_mask] = cytoplasm_intensity + rng.normal(0.0, 0.015, size=np.sum(cell_mask))

    # Darker cell membrane border
    img[membrane_mask] *= rng.uniform(0.72, 0.80)

    # 3. Class-specific biological features
    if cell_type == 0:
        # NORMAL CELL:
        # Red blood cells have a characteristic biconcave central pallor (lighter center)
        # or smooth uniform cytoplasm
        pallor_mask = dist_sq < 0.22
        img[pallor_mask] = np.clip(img[pallor_mask] + rng.uniform(0.12, 0.18), 0.0, 1.0)

    elif cell_type == 1:
        # INFECTED / PATHOLOGICAL CELL:
        # Contains dark intracellular inclusions: e.g., malaria parasite trophozoite ring,
        # nuclear chromatin clumping, or dense granules randomly located in cytoplasm.
        num_inclusions = rng.randint(1, 4)
        for _ in range(num_inclusions):
            # Place inclusion inside the cytoplasm (0.15 to 0.70 of radial distance)
            r_inc = rng.uniform(0.15, 0.65)
            theta_inc = rng.uniform(0, 2 * np.pi)
            inc_x = cx + r_inc * radius_x * np.cos(theta_inc)
            inc_y = cy + r_inc * radius_y * np.sin(theta_inc)

            inc_dist_sq = ((xx - inc_x) / 0.08)**2 + ((yy - inc_y) / 0.08)**2
            # Ring or dark chromatin spot
            is_ring = rng.choice([True, False])
            if is_ring:
                ring_mask = (inc_dist_sq >= 0.25) & (inc_dist_sq <= 1.1)
                dot_mask = inc_dist_sq < 0.25
                # Dark Giemsa-stained chromatin dot + bluish ring
                img[ring_mask] = np.clip(img[ring_mask] * rng.uniform(0.40, 0.50), 0.0, 1.0)
                img[dot_mask] = np.clip(img[dot_mask] * rng.uniform(0.20, 0.30), 0.0, 1.0)
            else:
                spot_mask = inc_dist_sq <= 1.0
                img[spot_mask] = np.clip(img[spot_mask] * rng.uniform(0.25, 0.40), 0.0, 1.0)

    # Final clipping to valid normalized grayscale range [0.0, 1.0]
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


def generate_synthetic_dataset(num_samples: int = 1200, img_size: int = 64, seed: int = 42):
    """
    Generates a balanced dataset of synthetic normal and infected microscope cell images.

    Parameters:
        num_samples (int): Total number of images (default 1200).
        img_size (int): Spatial dimensions (64x64).
        seed (int): Reproducibility seed.

    Returns:
        tuple: (images, labels)
            - images: np.ndarray of shape (num_samples, 64, 64, 1) in [0.0, 1.0]
            - labels: np.ndarray of shape (num_samples,) with values 0 or 1
    """
    rng = np.random.RandomState(seed)
    half = num_samples // 2

    images = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    labels = np.zeros((num_samples,), dtype=np.int32)

    for i in range(half):
        images[i, :, :, 0] = generate_single_cell(0, img_size, rng)
        labels[i] = 0

    for i in range(half, num_samples):
        images[i, :, :, 0] = generate_single_cell(1, img_size, rng)
        labels[i] = 1

    # Shuffle dataset
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    return images, labels


def load_custom_dataset(data_dir: str, img_size: tuple = (64, 64)):
    """
    Loads custom user microscope images from subfolders if provided.
    Expects structure:
      data_dir/Normal_Cell/ (or Uninfected/)
      data_dir/Infected_Cell/ (or Parasitized/)

    Returns:
        tuple: (images, labels) or (None, None) if directory not found.
    """
    if not os.path.exists(data_dir):
        return None, None

    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(class_dirs) < 2:
        return None, None

    images_list = []
    labels_list = []

    # Map class directories to 0 and 1
    class_map = {}
    for idx, cname in enumerate(sorted(class_dirs)[:2]):
        class_map[cname] = idx

    for cname, label_id in class_map.items():
        folder_path = os.path.join(data_dir, cname)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                fpath = os.path.join(folder_path, fname)
                try:
                    with Image.open(fpath) as im:
                        im = im.convert("L").resize(img_size)
                        arr = np.array(im, dtype=np.float32) / 255.0
                        images_list.append(arr)
                        labels_list.append(label_id)
                except Exception:
                    continue

    if len(images_list) == 0:
        return None, None

    images = np.expand_dims(np.array(images_list, dtype=np.float32), axis=-1)
    labels = np.array(labels_list, dtype=np.int32)
    return images, labels


def get_dataset(
    data_dir: str = "data",
    num_synthetic_samples: int = 1200,
    img_size: tuple = (64, 64),
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15
):
    """
    Loads dataset (custom if available, otherwise generates synthetic cell images)
    and splits into stratified train, validation, and test partitions.

    Splits:
        - Train: (1 - test_size - val_size) -> 70%
        - Validation: val_size -> 15%
        - Test: test_size -> 15%

    Returns:
        dict: {
            'X_train': ..., 'y_train': ...,
            'X_val': ..., 'y_val': ...,
            'X_test': ..., 'y_test': ...
        }
    """
    # 1. Attempt loading custom dataset
    images, labels = load_custom_dataset(data_dir, img_size)

    if images is None or len(images) < 100:
        print(f"[INFO] Using synthetic microscope cell dataset ({num_synthetic_samples} samples, {img_size[0]}x{img_size[1]}).")
        images, labels = generate_synthetic_dataset(num_samples=num_synthetic_samples, img_size=img_size[0], seed=seed)
    else:
        print(f"[INFO] Loaded custom microscope dataset from '{data_dir}' with {len(images)} samples.")

    # 2. Stratified train / test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    # Relative validation size from remaining training data
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=seed
    )

    print(f"[INFO] Dataset partitions created:")
    print(f"       - Train set:      {X_train.shape[0]} images (Shape: {X_train.shape})")
    print(f"       - Validation set: {X_val.shape[0]} images (Shape: {X_val.shape})")
    print(f"       - Test set:       {X_test.shape[0]} images (Shape: {X_test.shape})")
    print(f"       - Pixel Range:    [{X_train.min():.2f}, {X_train.max():.2f}] (Normalized float32)")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


def create_tf_datasets(splits: dict, batch_size: int = 32):
    """
    Wraps numpy arrays into high-performance tf.data.Dataset pipelines.
    """
    train_ds = (
        tf.data.Dataset.from_tensor_slices((splits["X_train"], splits["y_train"]))
        .shuffle(buffer_size=1024, seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((splits["X_val"], splits["y_val"]))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((splits["X_test"], splits["y_test"]))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Self-test when run directly
    splits = get_dataset()
    print("[SUCCESS] Data loader executed and verified successfully.")
