"""
models.py
=========
Neural Network Architectures for Microscope Image Classification.

Provides:
1. Fully Connected Network (FCN / Multi-Layer Perceptron):
   - Flattens 2D image into 1D vector (destroys local spatial pixel structure).
   - High parameter count in the first dense layer.
2. Convolutional Neural Network (CNN):
   - Preserves 2D topology using local receptive fields and weight sharing.
   - Extracts hierarchical spatial features (edges -> cell boundaries -> inclusions).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_fcn_model(input_shape: tuple = (64, 64, 1), num_classes: int = 2) -> models.Sequential:
    """
    Builds a classic Fully Connected Network (FCN / MLP).

    Architecture:
    -------------
    1. Input (64, 64, 1)
    2. Flatten: Unrolls the 64x64 grid into a 1D vector of length 4,096.
       *Key Concept*: This step discards all 2D geometric information.
       Pixels (x, y) and (x+1, y) are treated with the exact same independence
       as pixels on opposite corners of the image.
    3. Dense(128, ReLU): (4096 * 128) + 128 = 524,416 parameters in one layer!
    4. Dropout(0.3): Regularization to reduce overfitting.
    5. Dense(64, ReLU): (128 * 64) + 64 = 8,256 parameters.
    6. Dense(num_classes, Softmax): Output probability distribution.

    Parameters:
        input_shape (tuple): Spatial and channel dimensions (64, 64, 1).
        num_classes (int): Number of target classes (default 2).

    Returns:
        tf.keras.Model: Compiled or uncompiled FCN model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="fcn_input_image"),

        # Flatten layer: strips explicit 2D spatial arrangement
        layers.Flatten(name="fcn_flatten_to_1d"),

        # First dense layer (severe parameter explosion)
        layers.Dense(128, activation="relu", name="fcn_dense_128"),
        layers.Dropout(0.3, name="fcn_dropout_1"),

        # Second dense layer
        layers.Dense(64, activation="relu", name="fcn_dense_64"),

        # Output classification layer
        layers.Dense(num_classes, activation="softmax", name="fcn_output_probabilities")
    ], name="Fully_Connected_Network_FCN")

    return model


def build_cnn_model(input_shape: tuple = (64, 64, 1), num_classes: int = 2) -> models.Sequential:
    """
    Builds a 2D Convolutional Neural Network (CNN).

    Architecture:
    -------------
    1. Input (64, 64, 1)
    2. Conv2D(32, 3x3, ReLU, padding='same'):
       - 32 learnable spatial filters scanning small 3x3 local receptive fields.
       - Weight Sharing: Identical 3x3 weights slide across the entire image.
       - Parameter count: (3 * 3 * 1 + 1) * 32 = 320 parameters only!
       - Output: 32 feature maps of size (64, 64).
    3. MaxPooling2D(2, 2):
       - Downsamples spatial grid to (32, 32), retaining dominant activations
         and providing translation tolerance.
    4. Conv2D(64, 3x3, ReLU, padding='same'):
       - Learns combinations of low-level features (cellular rings, inclusions).
       - Parameter count: (3 * 3 * 32 + 1) * 64 = 18,496 parameters.
       - Output: 64 feature maps of size (32, 32).
    5. MaxPooling2D(2, 2):
       - Downsamples spatial grid to (16, 16).
    6. Flatten: Unrolls (16, 16, 64) -> 16,384 high-level feature values.
    7. Dense(64, ReLU): Fully connected classification head.
    8. Dropout(0.3): Regularization.
    9. Dense(num_classes, Softmax): Output probability distribution.

    Parameters:
        input_shape (tuple): Dimensions (64, 64, 1).
        num_classes (int): Number of target classes (default 2).

    Returns:
        tf.keras.Model: Uncompiled CNN model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="cnn_input_image"),

        # Block 1: Low-level spatial feature extraction (edges, boundaries)
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="cnn_conv1_32"),
        layers.MaxPooling2D(pool_size=(2, 2), name="cnn_pool1"),

        # Block 2: High-level feature combination (cellular textures, inclusions)
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="cnn_conv2_64"),
        layers.MaxPooling2D(pool_size=(2, 2), name="cnn_pool2"),

        # Classification Head
        layers.Flatten(name="cnn_flatten"),
        layers.Dense(64, activation="relu", name="cnn_dense_64"),
        layers.Dropout(0.3, name="cnn_dropout"),
        layers.Dense(num_classes, activation="softmax", name="cnn_output_probabilities")
    ], name="Convolutional_Neural_Network_CNN")

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Compiles model with Adam optimizer, Sparse Categorical Crossentropy, and Accuracy metric.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def print_comparison_summary():
    """Prints architectural and parameter comparison between FCN and CNN."""
    fcn = build_fcn_model()
    cnn = build_cnn_model()

    print("=" * 70)
    print("           MODEL ARCHITECTURAL & PARAMETER COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} | {'FCN (Dense/MLP)':<18} | {'CNN (Convolutional)':<18}")
    print("-" * 70)
    print(f"{'Input Representation':<30} | {'Flattened 1D (4096)':<18} | {'2D Grid (64x64x1)':<18}")
    print(f"{'Spatial Context':<30} | {'Destroyed':<18} | {'Preserved':<18}")
    print(f"{'Weight Sharing':<30} | {'None (1 weight/pixel)':<18} | {'Yes (Kernel sliding)':<18}")
    print(f"{'Translation Invariance':<30} | {'Very Low':<18} | {'High (via Pooling)':<18}")
    print(f"{'Layer 1 Feature Params':<30} | {fcn.layers[1].count_params():<18,d} | {cnn.layers[0].count_params():<18,d}")
    print(f"{'Total Parameters':<30} | {fcn.count_params():<18,d} | {cnn.count_params():<18,d}")
    print("=" * 70)


if __name__ == "__main__":
    print_comparison_summary()
