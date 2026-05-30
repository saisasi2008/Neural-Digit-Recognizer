"""
Advanced CNN Model for MNIST Digit Recognition.

Features:
  - Functional API with Residual (skip) connections
  - Squeeze-and-Excitation (SE) channel attention
  - Global Average Pooling
  - Built-in data augmentation layers
  - Ensemble model factory
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import Config


# ═══════════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════


def _se_block(x: tf.Tensor, filters: int, ratio: int = 8, prefix: str = "se") -> tf.Tensor:
    """
    Squeeze-and-Excitation block.
    Learns per-channel importance weights via global pooling → FC → sigmoid.
    """
    se = layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)
    se = layers.Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", name=f"{prefix}_fc1")(se)
    se = layers.Dense(filters, activation="sigmoid", kernel_initializer="he_normal", name=f"{prefix}_fc2")(se)
    se = layers.Reshape((1, 1, filters), name=f"{prefix}_reshape")(se)
    return layers.Multiply(name=f"{prefix}_mul")([x, se])


def _residual_block(x: tf.Tensor, filters: int, prefix: str = "res") -> tf.Tensor:
    """
    Residual block with two Conv2D layers and a skip connection.
    Uses 1×1 projection if channel dimensions differ.
    """
    shortcut = x

    out = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", name=f"{prefix}_conv1")(x)
    out = layers.BatchNormalization(name=f"{prefix}_bn1")(out)
    out = layers.ReLU(name=f"{prefix}_relu1")(out)

    out = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", name=f"{prefix}_conv2")(out)
    out = layers.BatchNormalization(name=f"{prefix}_bn2")(out)

    # Project shortcut if channel dimensions don't match
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal", name=f"{prefix}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{prefix}_proj_bn")(shortcut)

    out = layers.Add(name=f"{prefix}_add")([out, shortcut])
    out = layers.ReLU(name=f"{prefix}_relu_out")(out)
    return out


def _augmentation_block(x: tf.Tensor, cfg: Config) -> tf.Tensor:
    """Data augmentation layers (active only during training)."""
    x = layers.RandomRotation(cfg.augment_rotation, name="aug_rotation")(x)
    x = layers.RandomZoom(cfg.augment_zoom, name="aug_zoom")(x)
    x = layers.RandomTranslation(cfg.augment_translation, cfg.augment_translation, name="aug_translation")(x)
    x = layers.RandomContrast(cfg.augment_contrast, name="aug_contrast")(x)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════════════════════════


def create_model(cfg: Config = None) -> Model:
    """
    Build the advanced CNN model using the Functional API.

    Architecture:
        Input → Augmentation → [Conv+BN+ReLU → Residual → SE → MaxPool] × 3
        → GlobalAvgPool → Dense(256) → Dense(128) → Dense(10, softmax)

    Args:
        cfg: Configuration object. Uses defaults if None.

    Returns:
        Compiled-ready tf.keras.Model.
    """
    if cfg is None:
        cfg = Config()

    inputs = layers.Input(shape=cfg.input_shape, name="input_image")

    # ── Augmentation ────────────────────────────────────────────────────
    x = _augmentation_block(inputs, cfg)

    # ── Feature Extraction Blocks ───────────────────────────────────────
    for i, f in enumerate(cfg.filters):
        # Initial conv
        x = layers.Conv2D(f, (3, 3), padding="same", kernel_initializer="he_normal", name=f"conv_block{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_block{i+1}")(x)
        x = layers.ReLU(name=f"relu_block{i+1}")(x)

        # Residual connection
        if cfg.use_residual:
            x = _residual_block(x, f, prefix=f"res_block{i+1}")

        # Squeeze-and-Excitation attention
        if cfg.use_se_attention:
            x = _se_block(x, f, ratio=cfg.se_ratio, prefix=f"se_block{i+1}")

        # Downsample (skip for last block to preserve spatial info)
        if i < len(cfg.filters) - 1:
            x = layers.MaxPooling2D((2, 2), name=f"pool_block{i+1}")(x)

    # ── Classification Head ─────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    for i, (units, drop) in enumerate(zip(cfg.dense_units, cfg.dropout_rates)):
        x = layers.Dense(units, kernel_initializer="he_normal", name=f"head_dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"head_bn_{i+1}")(x)
        x = layers.ReLU(name=f"head_relu_{i+1}")(x)
        x = layers.Dropout(drop, name=f"head_dropout_{i+1}")(x)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="AdvancedMNIST_CNN")
    return model


def create_ensemble(cfg: Config) -> list:
    """
    Create multiple models with different random seeds for ensemble prediction.

    Args:
        cfg: Configuration object.

    Returns:
        List of tf.keras.Model instances.
    """
    models = []
    for i in range(cfg.ensemble_size):
        tf.random.set_seed(cfg.seed + i * 111)
        m = create_model(cfg)
        m._name = f"ensemble_model_{i}"
        models.append(m)
    return models


def get_last_conv_layer_name(model: Model) -> str:
    """Return the name of the last Conv2D layer (used for GradCAM)."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")