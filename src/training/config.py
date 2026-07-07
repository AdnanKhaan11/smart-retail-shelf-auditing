"""
src/training/config.py

RESPONSIBILITY
--------------
Defines the single, typed source of truth for every hyperparameter
and path used in a training run, and loads it from
configs/training_config.yaml. No hyperparameter should ever be
hardcoded directly inside train.py -- if train.py needs a value, it
should come from a TrainingConfig instance.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Typed configuration for a single RT-DETR fine-tuning run.

    Attributes:
        model_checkpoint: Hugging Face Hub identifier or local path
            for the pretrained RT-DETR checkpoint to fine-tune from.
        train_manifest: Path to the training split manifest.
        val_manifest: Path to the validation split manifest.
        image_size: Square image size (pixels). Must be positive.
        batch_size: Per-device training batch size. Must be positive.
        num_epochs: Number of full passes over the training set.
            Must be positive.
        learning_rate: Optimizer learning rate. Must be positive.
        output_dir: Directory checkpoints are written to.
        seed: Random seed for reproducibility.
        num_workers: Number of CPU worker processes for the
            DataLoader. Must be >= 0.
        persistent_workers: Keep DataLoader workers alive between
            epochs.
        pin_memory: Enables faster CPU->GPU transfer.
        weight_decay: L2 regularization strength for the optimizer.
            Must be >= 0.
        max_grad_norm: Gradient clipping threshold. Must be positive.
        scheduler: Learning rate schedule name, passed directly to
            transformers' lr_scheduler_type (e.g. "cosine", "linear").
            Must be a non-empty string.
        warmup_steps: Number of linear warmup steps. Must be >= 0.
        fp16: Request mixed-precision training. Automatically
            disabled by build_trainer() when no GPU is present.
        eval_every: Intended cadence (in epochs) for validation.
            NOTE: not yet wired into build_trainer() -- see that
            function's docstring.
        save_every: Same caveat as eval_every, for checkpoint saving.
        log_every: How often (in training steps) to log loss. Must
            be positive.
        resume_from_checkpoint: Optional explicit checkpoint path.
            If set, overrides auto-detection of the last checkpoint
            in output_dir. Leave null for normal auto-detect behavior.
    """

    model_checkpoint: str = "PekingU/rtdetr_v2_r50vd"
    train_manifest: str = "data/splits/train.json"
    val_manifest: str = "data/splits/val.json"
    image_size: int = 640
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    output_dir: str = "models/checkpoints"
    seed: int = 42
    num_workers: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.1
    scheduler: str = "cosine"
    warmup_steps: int = 500
    fp16: bool = True
    eval_every: int = 1
    save_every: int = 1
    log_every: int = 10
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Validate types and value ranges after construction.

        Raises:
            TypeError: If a numeric field isn't actually numeric
                (catches the classic YAML "1e-4" without a decimal
                point parsing as a string, not a float).
            ValueError: If a numeric field violates its required
                bound (positive, or non-negative where 0 is valid).
        """
        positive_int_fields = (
            "image_size",
            "batch_size",
            "num_epochs",
            "log_every",
            "eval_every",
            "save_every",
        )
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be an int, got {type(value).__name__}: {value!r}"
                )
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        non_negative_int_fields = ("num_workers", "warmup_steps")
        for field_name in non_negative_int_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be an int, got {type(value).__name__}: {value!r}"
                )
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value}")

        positive_float_fields = ("learning_rate", "max_grad_norm")
        for field_name in positive_float_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be numeric, got {type(value).__name__}: {value!r} "
                    f"(common cause: writing e.g. '1e-4' in YAML without a decimal point -- "
                    f"PyYAML parses that as a string; use '0.0001' or '1.0e-4' instead)"
                )
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        if not isinstance(self.weight_decay, (int, float)) or isinstance(
            self.weight_decay, bool
        ):
            raise TypeError(
                f"weight_decay must be numeric, got {type(self.weight_decay).__name__}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

        for field_name in ("persistent_workers", "pin_memory", "fp16"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(
                    f"{field_name} must be a bool, got {type(getattr(self, field_name)).__name__}"
                )

        if not isinstance(self.scheduler, str) or len(self.scheduler) == 0:
            raise TypeError(
                f"scheduler must be a non-empty string, got {self.scheduler!r}"
            )

        if self.resume_from_checkpoint is not None and not isinstance(
            self.resume_from_checkpoint, str
        ):
            raise TypeError(
                f"resume_from_checkpoint must be null or a string path, "
                f"got {type(self.resume_from_checkpoint).__name__}"
            )


def load_config(config_path: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        TrainingConfig: Configuration object populated from the YAML
        file. Any fields missing from the YAML automatically use the
        dataclass default values.

    Raises:
        FileNotFoundError: If the configuration file does not exist,
            or if config_path points to a directory rather than a file.
        ValueError: If the YAML file's top-level content isn't a
            mapping, or if it contains a key that doesn't match any
            TrainingConfig field.
        TypeError: Propagated from TrainingConfig.__post_init__ if a
            numeric field has the wrong type.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration path is not a file: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        raw_dict = yaml.safe_load(file) or {}

    if not isinstance(raw_dict, dict):
        raise ValueError(
            f"Expected {config_path} to contain a YAML mapping (key: value pairs), "
            f"got {type(raw_dict).__name__} instead."
        )

    try:
        config = TrainingConfig(**raw_dict)
    except TypeError as exc:
        raise ValueError(
            f"Failed to build TrainingConfig from {config_path}: {exc}. "
            f"Check for a misspelled or unexpected key in the YAML file."
        ) from exc

    return config
