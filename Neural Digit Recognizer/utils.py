"""
Utility functions for visualization, GradCAM, and plotting.
All plots are saved to the results directory and optionally displayed.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os

# Use non-interactive backend when saving to file
matplotlib.use("Agg")

# ── Global plot styling ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
    "font.family": "sans-serif",
})

COLORS = {
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "orange": "#d29922",
    "purple": "#bc8cff",
    "gradient": ["#58a6ff", "#bc8cff", "#f778ba", "#ff7b72", "#d29922",
                  "#3fb950", "#79c0ff", "#d2a8ff", "#ffa657", "#7ee787"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  GradCAM
# ═══════════════════════════════════════════════════════════════════════════════


def compute_gradcam(model: tf.keras.Model, image: np.ndarray, class_idx: int = None,
                    last_conv_layer: str = None) -> np.ndarray:
    """
    Compute Gradient-weighted Class Activation Map (Grad-CAM).

    Args:
        model: Trained Keras model.
        image: Input image array of shape (H, W, 1) or (1, H, W, 1).
        class_idx: Target class index. If None, uses predicted class.
        last_conv_layer: Name of the last conv layer. Auto-detected if None.

    Returns:
        Heatmap array of shape (H, W) with values in [0, 1].
    """
    if last_conv_layer is None:
        from model import get_last_conv_layer_name
        last_conv_layer = get_last_conv_layer_name(model)

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    # Build sub-model that outputs both conv activations and predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image, training=False)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Gradient of the predicted class w.r.t. conv layer output
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    # Resize to input dimensions
    heatmap = heatmap.numpy()
    heatmap = np.uint8(255 * heatmap)
    # Resize using tf.image
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (image.shape[1], image.shape[2])
    ).numpy().squeeze()
    heatmap_resized = heatmap_resized / 255.0
    return heatmap_resized


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay GradCAM heatmap on grayscale image. Returns RGB array."""
    # Normalize image to [0, 1]
    img = image.squeeze()
    if img.max() > 1.0:
        img = img / 255.0

    # Create RGB from grayscale
    img_rgb = np.stack([img] * 3, axis=-1)

    # Apply colormap to heatmap
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]

    # Blend
    overlaid = (1 - alpha) * img_rgb + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════════


def plot_sample_predictions(model, x_test, y_test, save_dir="results", num_samples=15):
    """Plot sample predictions with confidence bars and GradCAM overlays."""
    predictions = model.predict(x_test[:num_samples], verbose=0)
    rows = 3
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle("Sample Predictions with GradCAM", fontsize=16, fontweight="bold", color=COLORS["accent"])

    for i in range(num_samples):
        ax = axes[i // cols, i % cols]
        pred_label = np.argmax(predictions[i])
        true_label = y_test[i]
        confidence = predictions[i][pred_label] * 100

        # Compute GradCAM overlay
        try:
            heatmap = compute_gradcam(model, x_test[i])
            overlaid = overlay_gradcam(x_test[i], heatmap)
            ax.imshow(overlaid)
        except Exception:
            ax.imshow(x_test[i].squeeze(), cmap="gray")

        color = COLORS["green"] if pred_label == true_label else COLORS["red"]
        status = "✓" if pred_label == true_label else "✗"
        ax.set_title(f"{status} Pred: {pred_label} ({confidence:.1f}%)", color=color, fontsize=9, fontweight="bold")
        ax.set_xlabel(f"True: {true_label}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "sample_predictions.png"), bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_dir}/sample_predictions.png")


def plot_confidence_bars(predictions: np.ndarray, true_label: int, save_path: str = None):
    """Plot a horizontal bar chart of prediction confidences for a single sample."""
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = []
    pred_label = np.argmax(predictions)
    for i in range(10):
        if i == pred_label and i == true_label:
            colors.append(COLORS["green"])
        elif i == pred_label:
            colors.append(COLORS["red"])
        elif i == true_label:
            colors.append(COLORS["orange"])
        else:
            colors.append("#30363d")

    bars = ax.barh(range(10), predictions * 100, color=colors, height=0.6, edgecolor="#21262d")
    ax.set_yticks(range(10))
    ax.set_yticklabels([str(i) for i in range(10)])
    ax.set_xlabel("Confidence (%)")
    ax.set_title(f"Prediction Confidence  |  True: {true_label}  Pred: {pred_label}",
                 fontweight="bold", color=COLORS["accent"])
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add percentage labels
    for bar, val in zip(bars, predictions * 100):
        if val > 2:
            ax.text(val - 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                    va="center", ha="right", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()