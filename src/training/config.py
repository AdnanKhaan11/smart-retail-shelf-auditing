"""
src.training.config

RESPONSIBILITY
--------------
Defines the single, typed source of truth for every hyperparameter
and path used in a training run, and loads it from
configs/training_config.yaml. No hyperparameter should ever be
hardcoded directly inside train.py — if train.py needs a value, it
should come from a TrainingConfig instance.

PURPOSE
-------
Externalizing configuration is what makes your experiments in
experiments/experiment_log.md (see docs/SDP.md Section 4, Day 9)
actually reproducible and comparable — "run 3 used lr=5e-5,
batch_size=16" is only a meaningful sentence if that value came from
one config file you can diff against run 2's config file, not from a
number buried inside a Python script you may have since edited.

ARCHITECTURE NOTES
-------------------
    • TrainingConfig below is intentionally a plain, flat dataclass
      with sane defaults — it is fully implemented as your reference
      pattern (same approach as backend/schemas.py's BoundingBox).
    • load_config() is what is NOT implemented — reading and
      validating the actual YAML file is your job.
    • Keep this config SEPARATE from configs/inference_config.yaml
      (used by backend/). Training-time config (epochs, learning
      rate, augmentation) and inference-time config (thresholds,
      model path) have different lifecycles and different owners in
      a real team, even though both eventually point at "the same"
      model.

ASCII FLOW DIAGRAM
-------------------
    configs/training_config.yaml
            |
            v
    load_config(path)          <- YOU implement this
            |
            v
    TrainingConfig(...)         <- already implemented (reference pattern)
            |
            v
    src/training/train.py consumes config.* fields directly

TODO
----
    - [ ] Implement load_config():
          1. Read the YAML file at config_path using PyYAML
             (yaml.safe_load)
          2. Construct and return a TrainingConfig, overriding the
             dataclass defaults with whatever keys are present in
             the YAML file
          3. Raise a clear FileNotFoundError if config_path doesn't
             exist, rather than letting a cryptic yaml/IO error
             surface
    - [ ] Add any fields you discover you need once you implement
          src/data/ and src/training/train.py (e.g. augmentation
          toggles, gradient_accumulation_steps) — this dataclass is
          expected to grow as the project does.

HINTS
-----
    - `yaml.safe_load(f)` returns a plain dict — use `**` unpacking
      into TrainingConfig(**raw_dict) if the YAML keys exactly match
      the dataclass field names (recommended for simplicity).
    - Keep configs/training_config.yaml under version control (it's
      small and text-based) — unlike models/checkpoints/, which is
      gitignored.

COMMON MISTAKES
----------------
    - Editing hyperparameter values directly in train.py "just this
      once" instead of updating the YAML file — this is exactly the
      kind of drift that makes experiments impossible to reproduce
      later.
    - Silently swallowing a missing/malformed YAML file and falling
      back to defaults — fail loudly instead, so a typo in the YAML
      doesn't quietly train with the wrong settings.

BEST PRACTICES
---------------
    - Every training run should log its resolved TrainingConfig
      (e.g. as a JSON dump next to the saved checkpoint) so you can
      always answer "what config produced this specific model file."

LEARNING NOTES
--------------
Notice that TrainingConfig itself is fully implemented — designing a
clean, flat config schema is a good pattern to copy, not something
you need to "discover" from scratch. Your job is the YAML-loading
plumbing around it, not the shape of the config itself.

REFERENCES
----------
    - https://pyyaml.org/wiki/PyYAMLDocumentation
    - https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Typed configuration for a single RT-DETR fine-tuning run.

    Fully implemented as your reference pattern — copy this flat,
    explicit style for any config you add later in this project.

    Attributes:
        model_checkpoint: Hugging Face Hub identifier or local path
            for the pretrained RT-DETR checkpoint to fine-tune from.
        train_manifest: Path to the training split manifest produced
            in docs/SDP.md Day 6 (data/splits/train.json).
        val_manifest: Path to the validation split manifest
            (data/splits/val.json).
        image_size: Square image size (pixels) fed to the image
            processor during preprocessing.
        batch_size: Per-device training batch size.
        num_epochs: Number of full passes over the training set.
        learning_rate: Optimizer learning rate.
        output_dir: Directory checkpoints are written to
            (models/checkpoints/).
        seed: Random seed for reproducibility across data shuffling,
            augmentation, and model initialization.
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


def load_config(config_path: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.

    Args:
        config_path: Path to a YAML file, expected to be
            configs/training_config.yaml relative to the project
            root, but accepted as a plain string so this function
            stays testable with any path.

    Returns:
        A TrainingConfig populated from the YAML file's contents,
        falling back to the dataclass defaults for any field the
        YAML file does not specify.

    Raises:
        NotImplementedError: Always, until implemented.
        FileNotFoundError: Should be raised (once implemented) if
            config_path does not point to an existing file.
    """
    raise NotImplementedError("load_config() is not implemented yet")
