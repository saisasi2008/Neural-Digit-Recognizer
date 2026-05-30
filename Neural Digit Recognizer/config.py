"""
Centralized configuration for the MNIST Digit Recognizer.
All hyperparameters, paths, and training settings in one place.
"""

from dataclasses import dataclass, field
from typing import Optional
import argparse
import os
import json
from datetime import datetime


@dataclass
class Config:
    """Master configuration for training, evaluation, and deployment."""

    # ── Model Architecture ──────────────────────────────────────────────
    input_shape: tuple = (28, 28, 1)
    num_classes: int = 10
    filters: list = field(default_factory=lambda: [32, 64, 128])
    dense_units: list = field(default_factory=lambda: [256, 128])
    dropout_rates: list = field(default_factory=lambda: [0.5, 0.3])
    use_residual: bool = True
    use_se_attention: bool = True
    se_ratio: int = 8  # Squeeze-and-Excitation reduction ratio

    # ── Data Augmentation ───────────────────────────────────────────────
    augment_rotation: float = 0.1
    augment_zoom: float = 0.1
    augment_translation: float = 0.1
    augment_contrast: float = 0.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2

    # ── Training ────────────────────────────────────────────────────────
    epochs: int = 20
    batch_size: int = 64
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    val_split: float = 0.2
    seed: int = 42

    # ── Callbacks ───────────────────────────────────────────────────────
    early_stop_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    # ── Ensemble ────────────────────────────────────────────────────────
    use_ensemble: bool = False
    ensemble_size: int = 3

    # ── Paths ───────────────────────────────────────────────────────────
    model_dir: str = "saved_model"
    results_dir: str = "results"
    experiments_dir: str = "experiments"
    tensorboard_dir: str = "logs/tensorboard"

    # ── Export ───────────────────────────────────────────────────────────
    export_tflite: bool = True
    export_onnx: bool = True
    tflite_quantize: bool = True

    def __post_init__(self):
        """Create output directories."""
        for d in [self.model_dir, self.results_dir, self.experiments_dir, self.tensorboard_dir]:
            os.makedirs(d, exist_ok=True)

    def save(self, path: Optional[str] = None) -> str:
        """Save config to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.experiments_dir, f"config_{timestamp}.json")
        data = {k: v for k, v in self.__dict__.items()}
        # Convert tuples/lists for JSON serialization
        data["input_shape"] = list(data["input_shape"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        data["input_shape"] = tuple(data["input_shape"])
        return cls(**data)


def parse_args() -> Config:
    """Parse command-line arguments and return a Config object."""
    parser = argparse.ArgumentParser(
        description="🔢 Advanced MNIST Digit Recognizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Train with defaults
  python main.py --epochs 30 --batch-size 128
  python main.py --use-ensemble --ensemble-size 5
  python main.py --config experiments/config.json
        """,
    )

    parser.add_argument("--config", type=str, help="Load config from JSON file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-ensemble", action="store_true", default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections")
    parser.add_argument("--no-se", action="store_true", help="Disable SE attention")
    parser.add_argument("--no-mixup", action="store_true", help="Disable mixup augmentation")
    parser.add_argument("--label-smoothing", type=float, default=None)

    args = parser.parse_args()

    # Load base config
    if args.config:
        cfg = Config.from_json(args.config)
    else:
        cfg = Config()

    # Override with CLI args
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.initial_lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.use_ensemble is not None:
        cfg.use_ensemble = args.use_ensemble
    if args.ensemble_size is not None:
        cfg.ensemble_size = args.ensemble_size
    if args.no_residual:
        cfg.use_residual = False
    if args.no_se:
        cfg.use_se_attention = False
    if args.no_mixup:
        cfg.use_mixup = False
    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing

    return cfg
