"""
Advanced MNIST Digit Recognizer — Main Training Pipeline.

Usage:
  python main.py                              # Train with defaults
  python main.py --epochs 30 --batch-size 128
  python main.py --use-ensemble --ensemble-size 5
  python main.py --no-residual --no-se        # Ablation study
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

from config import Config, parse_args
from model import create_model, create_ensemble
from evaluate import run_full_evaluation
from export import export_all
from utils import plot_sample_predictions


# ═══════════════════════════════════════════════════════════════════════════════
#  Seed Everything
# ═══════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed: int):
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(cfg: Config):
    """Load MNIST and split into train/val/test with preprocessing."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and reshape for CNN
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # Stratified-ish split with shuffling
    np.random.seed(cfg.seed)
    indices = np.random.permutation(len(train_images))
    train_size = int((1 - cfg.val_split) * len(train_images))

    x_train = train_images[indices[:train_size]]
    y_train = train_labels[indices[:train_size]]
    x_val = train_images[indices[train_size:]]
    y_val = train_labels[indices[train_size:]]

    return x_train, y_train, x_val, y_val, test_images, test_labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Mixup Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(x: np.ndarray, y: np.ndarray, alpha: float = 0.2):
    """Apply mixup augmentation: blend random pairs of samples."""
    batch_size = len(x)
    lam = np.random.beta(alpha, alpha, batch_size).astype("float32")
    lam_x = lam.reshape(-1, 1, 1, 1)
    lam_y = lam.reshape(-1, 1)

    indices = np.random.permutation(batch_size)

    x_mixed = lam_x * x + (1 - lam_x) * x[indices]
    y_onehot = tf.keras.utils.to_categorical(y, 10).astype("float32")
    y_mixed = lam_y * y_onehot + (1 - lam_y) * y_onehot[indices]

    return x_mixed, y_mixed


def create_mixup_dataset(x: np.ndarray, y: np.ndarray, cfg: Config) -> tf.data.Dataset:
    """Create a tf.data.Dataset with optional mixup augmentation."""
    if cfg.use_mixup:
        x_mixed, y_mixed = mixup_data(x, y, cfg.mixup_alpha)
        dataset = tf.data.Dataset.from_tensor_slices((x_mixed, y_mixed))
    else:
        y_onehot = tf.keras.utils.to_categorical(y, 10).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((x, y_onehot))

    dataset = (dataset
               .shuffle(buffer_size=len(x), seed=cfg.seed)
               .batch(cfg.batch_size)
               .prefetch(tf.data.AUTOTUNE))
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  Callbacks & LR Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_annealing_schedule(epoch, lr, cfg: Config):
    """Cosine annealing with warm restarts."""
    restart_period = 10  # Restart every 10 epochs
    epoch_in_cycle = epoch % restart_period
    return cfg.min_lr + 0.5 * (cfg.initial_lr - cfg.min_lr) * (
        1 + np.cos(np.pi * epoch_in_cycle / restart_period)
    )


def get_callbacks(cfg: Config) -> list:
    """Build the callbacks suite."""
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.model_dir, "mnist_advanced.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg.tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
        ),
        # Cosine annealing LR
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_schedule(epoch, lr, cfg),
            verbose=0,
        ),
    ]
    return callbacks


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════

def compile_model(model: tf.keras.Model, cfg: Config):
    """Compile model with optimizer and label-smoothed loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.initial_lr)

    # Use categorical crossentropy (since we use one-hot labels for mixup)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def train_single_model(model, train_ds, x_val, y_val, cfg: Config, model_name: str = "Model"):
    """Train a single model and return history."""
    print(f"\n{'─' * 60}")
    print(f"  🚀 Training {model_name}")
    print(f"{'─' * 60}")

    # Val labels need to be one-hot for categorical crossentropy
    y_val_onehot = tf.keras.utils.to_categorical(y_val, 10)

    history = model.fit(
        train_ds,
        epochs=cfg.epochs,
        validation_data=(x_val, y_val_onehot),
        callbacks=get_callbacks(cfg),
        verbose=1,
    )
    return history


def ensemble_predict(models: list, x: np.ndarray) -> np.ndarray:
    """Average predictions from multiple models."""
    predictions = [m.predict(x, verbose=0) for m in models]
    return np.mean(predictions, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    # ── Parse config ────────────────────────────────────────────────────
    cfg = parse_args()
    set_all_seeds(cfg.seed)

    print("\n" + "═" * 60)
    print("  🔢  ADVANCED MNIST DIGIT RECOGNIZER")
    print("═" * 60)
    print(f"  Epochs:          {cfg.epochs}")
    print(f"  Batch Size:      {cfg.batch_size}")
    print(f"  Learning Rate:   {cfg.initial_lr}")
    print(f"  Label Smoothing: {cfg.label_smoothing}")
    print(f"  Mixup:           {'✅' if cfg.use_mixup else '❌'}")
    print(f"  Residual:        {'✅' if cfg.use_residual else '❌'}")
    print(f"  SE Attention:    {'✅' if cfg.use_se_attention else '❌'}")
    print(f"  Ensemble:        {'✅ (' + str(cfg.ensemble_size) + ' models)' if cfg.use_ensemble else '❌'}")
    print("═" * 60)

    # ── Save config ─────────────────────────────────────────────────────
    config_path = cfg.save()
    print(f"\n  💾 Config saved: {config_path}")

    # ── Load data ───────────────────────────────────────────────────────
    print("\n  📥 Loading MNIST dataset...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess(cfg)
    print(f"     Training:   {len(x_train):,} samples")
    print(f"     Validation: {len(x_val):,} samples")
    print(f"     Test:       {len(x_test):,} samples")

    # ── Create dataset ──────────────────────────────────────────────────
    train_ds = create_mixup_dataset(x_train, y_train, cfg)

    # ── Train ───────────────────────────────────────────────────────────
    if cfg.use_ensemble:
        models = create_ensemble(cfg)
        histories = []
        for i, m in enumerate(models):
            m = compile_model(m, cfg)
            # Re-create mixup dataset per model for diversity
            set_all_seeds(cfg.seed + i * 111)
            ds = create_mixup_dataset(x_train, y_train, cfg)
            h = train_single_model(m, ds, x_val, y_val, cfg, f"Ensemble Member {i+1}/{cfg.ensemble_size}")
            histories.append(h)
            models[i] = m

        # Ensemble evaluation
        print("\n  🏆 Ensemble Evaluation:")
        y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
        ensemble_preds = ensemble_predict(models, x_test)
        ensemble_acc = np.mean(np.argmax(ensemble_preds, axis=1) == y_test) * 100
        print(f"     Ensemble Test Accuracy: {ensemble_acc:.2f}%")

        # Use the best single model for exports and visualizations
        best_idx = np.argmax([max(h.history["val_accuracy"]) for h in histories])
        model = models[best_idx]
        history = histories[best_idx]
        print(f"     Best single model: #{best_idx + 1}")
    else:
        model = create_model(cfg)
        model = compile_model(model, cfg)
        model.summary()
        history = train_single_model(model, train_ds, x_val, y_val, cfg, "AdvancedMNIST_CNN")

    # ── Test ────────────────────────────────────────────────────────────
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
    test_loss, test_acc = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f"\n  🏆 Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  📉 Test Loss:     {test_loss:.4f}")

    # ── Save model ──────────────────────────────────────────────────────
    model_path = os.path.join(cfg.model_dir, "mnist_advanced.keras")
    model.save(model_path)
    print(f"\n  💾 Model saved: {model_path}")

    # ── Full evaluation ─────────────────────────────────────────────────
    run_full_evaluation(model, x_test, y_test, history, cfg.results_dir)

    # ── Sample predictions ──────────────────────────────────────────────
    plot_sample_predictions(model, x_test, y_test, cfg.results_dir)

    # ── Export ──────────────────────────────────────────────────────────
    export_all(model, x_test, cfg.model_dir, cfg.tflite_quantize, cfg.export_onnx)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "═" * 60)
    print("  ✅  TRAINING COMPLETE")
    print(f"  ⏱️  Total time: {elapsed / 60:.1f} minutes")
    print(f"  🏆  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  📂  Results: {cfg.results_dir}/")
    print(f"  📂  Models:  {cfg.model_dir}/")
    print(f"  📊  TensorBoard: tensorboard --logdir {cfg.tensorboard_dir}")
    print(f"  🌐  Web Demo: python app.py")
    print("═" * 60)

    # ── Save experiment log ─────────────────────────────────────────────
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v if not isinstance(v, tuple) else list(v)
                   for k, v in cfg.__dict__.items()},
        "results": {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_val_loss": float(min(history.history["val_loss"])),
            "total_epochs_trained": len(history.history["loss"]),
            "training_time_seconds": elapsed,
        },
        "model_params": model.count_params(),
    }
    log_path = os.path.join(cfg.experiments_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"\n  📝 Experiment log: {log_path}")


if __name__ == "__main__":
    main()