"""
Comprehensive evaluation module for the MNIST Digit Recognizer.

Generates:
  - Confusion matrix heatmap
  - Per-class classification report
  - Training history curves (loss + accuracy)
  - Confidence distribution analysis
  - Misclassification gallery with GradCAM
  - GradCAM saliency map grid
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")

from sklearn.metrics import confusion_matrix, classification_report
from utils import compute_gradcam, overlay_gradcam, COLORS


def run_full_evaluation(model, x_test, y_test, history=None, save_dir="results"):
    """Run all evaluation steps and save results."""
    os.makedirs(save_dir, exist_ok=True)
    predictions = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    pred_confidence = np.max(predictions, axis=1)

    print("\n" + "═" * 60)
    print("  📊  COMPREHENSIVE EVALUATION REPORT")
    print("═" * 60)

    # 1. Classification Report
    _print_classification_report(y_test, pred_labels)

    # 2. Confusion Matrix
    _plot_confusion_matrix(y_test, pred_labels, save_dir)

    # 3. Training History
    if history is not None:
        _plot_training_history(history, save_dir)

    # 4. Confidence Distribution
    _plot_confidence_distribution(pred_confidence, y_test, pred_labels, save_dir)

    # 5. Misclassification Gallery
    _plot_misclassification_gallery(model, x_test, y_test, predictions, save_dir)

    # 6. GradCAM Grid
    _plot_gradcam_grid(model, x_test, y_test, save_dir)

    # 7. Per-digit accuracy bar chart
    _plot_per_digit_accuracy(y_test, pred_labels, save_dir)

    print(f"\n  📂  All plots saved to: {save_dir}/")
    print("═" * 60)


def _print_classification_report(y_true, y_pred):
    """Print a formatted classification report."""
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
    print(f"\n  📋 Classification Report:\n")
    print(report)


def _plot_confusion_matrix(y_true, y_pred, save_dir):
    """Generate a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors="#c9d1d9")
    cbar.outline.set_edgecolor("#30363d")

    # Labels
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(10)], fontsize=12)
    ax.set_yticklabels([str(i) for i in range(10)], fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", color=COLORS["accent"], pad=15)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > thresh else "#c9d1d9"
            fontweight = "bold" if i == j else "normal"
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color=color, fontsize=10, fontweight=fontweight)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: confusion_matrix.png")


def _plot_training_history(history, save_dir):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs = range(1, len(history.history["loss"]) + 1)

    # ── Loss ────────────────────────────────────────────────────────────
    ax1.plot(epochs, history.history["loss"], color=COLORS["accent"],
             linewidth=2, label="Training Loss", marker="o", markersize=4)
    ax1.plot(epochs, history.history["val_loss"], color=COLORS["red"],
             linewidth=2, label="Validation Loss", marker="s", markersize=4, linestyle="--")
    ax1.fill_between(epochs, history.history["loss"], alpha=0.1, color=COLORS["accent"])
    ax1.fill_between(epochs, history.history["val_loss"], alpha=0.1, color=COLORS["red"])
    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold", color=COLORS["accent"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(framealpha=0.3, edgecolor="#30363d")
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ────────────────────────────────────────────────────────
    ax2.plot(epochs, [a * 100 for a in history.history["accuracy"]], color=COLORS["green"],
             linewidth=2, label="Training Accuracy", marker="o", markersize=4)
    ax2.plot(epochs, [a * 100 for a in history.history["val_accuracy"]], color=COLORS["purple"],
             linewidth=2, label="Validation Accuracy", marker="s", markersize=4, linestyle="--")
    ax2.fill_between(epochs, [a * 100 for a in history.history["accuracy"]], alpha=0.1, color=COLORS["green"])
    ax2.fill_between(epochs, [a * 100 for a in history.history["val_accuracy"]], alpha=0.1, color=COLORS["purple"])
    ax2.set_title("Accuracy Curves", fontsize=14, fontweight="bold", color=COLORS["accent"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(framealpha=0.3, edgecolor="#30363d")
    ax2.grid(True, alpha=0.3)

    # Add best epoch annotation
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_acc = max(history.history["val_accuracy"]) * 100
    ax2.annotate(f"Best: {best_acc:.2f}% (Epoch {best_epoch})",
                 xy=(best_epoch, best_acc), fontsize=9, fontweight="bold",
                 color=COLORS["green"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5),
                 xytext=(best_epoch + 1, best_acc - 2))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: training_history.png")


def _plot_confidence_distribution(confidence, y_true, y_pred, save_dir):
    """Plot confidence distribution for correct vs. incorrect predictions."""
    correct_mask = y_true == y_pred
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(correct_conf, bins=50, alpha=0.7, color=COLORS["green"],
            label=f"Correct ({len(correct_conf):,})", edgecolor="#0d1117", linewidth=0.5)
    if len(incorrect_conf) > 0:
        ax.hist(incorrect_conf, bins=50, alpha=0.7, color=COLORS["red"],
                label=f"Incorrect ({len(incorrect_conf):,})", edgecolor="#0d1117", linewidth=0.5)

    ax.set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold", color=COLORS["accent"])
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Correct: μ={correct_conf.mean():.4f}, σ={correct_conf.std():.4f}"
    if len(incorrect_conf) > 0:
        stats_text += f"\nIncorrect: μ={incorrect_conf.mean():.4f}, σ={incorrect_conf.std():.4f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", color="#8b949e",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d"))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: confidence_distribution.png")


def _plot_misclassification_gallery(model, x_test, y_test, predictions, save_dir, num_samples=20):
    """Show the most confident wrong predictions with GradCAM overlays."""
    pred_labels = np.argmax(predictions, axis=1)
    pred_confidence = np.max(predictions, axis=1)

    # Find misclassified samples, sorted by confidence (most confident errors first)
    wrong_mask = pred_labels != y_test
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("  ⚠️  No misclassifications found — perfect accuracy!")
        return

    wrong_conf = pred_confidence[wrong_indices]
    sorted_wrong = wrong_indices[np.argsort(wrong_conf)[::-1]][:num_samples]

    n = min(num_samples, len(sorted_wrong))
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.suptitle("Most Confident Misclassifications", fontsize=16, fontweight="bold", color=COLORS["red"])

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            i = sorted_wrong[idx]
            try:
                heatmap = compute_gradcam(model, x_test[i])
                overlaid = overlay_gradcam(x_test[i], heatmap)
                ax.imshow(overlaid)
            except Exception:
                ax.imshow(x_test[i].squeeze(), cmap="gray")

            pred = pred_labels[i]
            true = y_test[i]
            conf = pred_confidence[i] * 100
            ax.set_title(f"✗ Pred: {pred} ({conf:.1f}%)", color=COLORS["red"], fontsize=9, fontweight="bold")
            ax.set_xlabel(f"True: {true}", fontsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLORS["red"])
                spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "misclassifications.png"), bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: misclassifications.png")


def _plot_gradcam_grid(model, x_test, y_test, save_dir, num_samples=10):
    """Show GradCAM saliency maps for correctly classified samples (one per digit)."""
    predictions = model.predict(x_test[:1000], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.suptitle("GradCAM Saliency Maps by Digit", fontsize=16, fontweight="bold", color=COLORS["accent"])

    for digit in range(10):
        # Find a correctly classified example of this digit
        mask = (y_test[:1000] == digit) & (pred_labels == digit)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        idx = indices[0]

        # Original image
        axes[0, digit].imshow(x_test[idx].squeeze(), cmap="gray")
        axes[0, digit].set_title(f"Digit {digit}", fontsize=10, fontweight="bold", color=COLORS["gradient"][digit])
        axes[0, digit].set_xticks([])
        axes[0, digit].set_yticks([])

        # GradCAM overlay
        try:
            heatmap = compute_gradcam(model, x_test[idx])
            overlaid = overlay_gradcam(x_test[idx], heatmap)
            axes[1, digit].imshow(overlaid)
        except Exception:
            axes[1, digit].imshow(x_test[idx].squeeze(), cmap="gray")
        axes[1, digit].set_xticks([])
        axes[1, digit].set_yticks([])

    axes[0, 0].set_ylabel("Original", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("GradCAM", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradcam_grid.png"), bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: gradcam_grid.png")


def _plot_per_digit_accuracy(y_true, y_pred, save_dir):
    """Bar chart showing accuracy per digit class."""
    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = []
    counts = []
    for digit in range(10):
        mask = y_true == digit
        correct = np.sum(y_pred[mask] == digit)
        total = np.sum(mask)
        acc = correct / total * 100 if total > 0 else 0
        accuracies.append(acc)
        counts.append(total)

    bars = ax.bar(range(10), accuracies, color=COLORS["gradient"], edgecolor="#0d1117",
                  linewidth=0.5, width=0.7)

    # Add labels on bars
    for bar, acc, count in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c9d1d9")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"n={count}", ha="center", va="center", fontsize=7, color="white")

    ax.set_xticks(range(10))
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Digit Accuracy", fontsize=14, fontweight="bold", color=COLORS["accent"])
    ax.set_ylim(95, 100.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_digit_accuracy.png"), bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: per_digit_accuracy.png")
