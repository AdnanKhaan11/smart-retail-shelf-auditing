"""
src.training.train

RESPONSIBILITY
--------------
Orchestrates one full RT-DETR fine-tuning run: build the dataset,
build the model, configure the Hugging Face Trainer, run training,
resume automatically from the last checkpoint if one exists, and run
a standalone, verified evaluation pass at the end.

CHECKPOINT STORAGE -- IMPORTANT LESSON LEARNED
--------------------------------------------------
config.output_dir should be a LOCAL disk path (e.g. /content/checkpoints
on Colab), NOT a Drive-mounted path. Writing Trainer checkpoints
directly to Drive was tried and failed: across an entire 25-epoch run,
model.safetensors and optimizer.pt (the two largest files) were
silently missing from every single checkpoint, while small metadata
files (config.json, scheduler.pt, etc.) always succeeded. Drive's
mount does not reliably support the write-then-rename pattern Trainer
uses for large files. Instead, set config.drive_sync_dir -- after each
local checkpoint completes, SyncCheckpointToDriveCallback copies the
already-verified-complete local checkpoint folder to Drive as a
simple file copy, which does not have this problem.

ARCHITECTURE NOTES
-------------------
    • build_datasets(config) loads its own image processor internally
      (via get_image_processor(), same as build_model()) rather than
      taking one as a parameter -- this keeps the function's public
      signature matching its own docstring (config only), since
      loading an image processor config is cheap and stateless.
    • build_datasets() uses with_transform(), NOT map(), for the
      preprocessing step. map() would permanently WRITE
      preprocess_batch()'s output into the dataset's Arrow-backed
      storage -- and Arrow silently converts torch.Tensor objects
      into plain nested Python lists on write, which breaks RT-DETR's
      forward pass deep inside the model. with_transform() instead
      applies the function fresh, on every single access, so the
      tensors it returns stay real tensors. Augmentation is ALSO done
      inside this same lazy transform (not via a separate map() call)
      -- augmenting via map() previously hit a real ArrowMemoryError
      (a 2GB single-chunk limit) when writing 4000+ augmented images
      to Arrow storage. Doing both augmentation and preprocessing
      lazily, per-batch, avoids writing any image data to Arrow at
      all.
    • Two different "val dataset" representations exist on purpose:
      the Trainer-ready one (fully processor-encoded, used only for
      Trainer's built-in eval_loss tracking) and a separate raw
      (augmented-but-not-processor-encoded) one built by
      _build_raw_eval_dataset(), used for the real, verified
      mAP/precision/recall/F1 numbers via evaluate_on_dataset().
    • fp16 is only meaningful (and only valid) on a CUDA GPU --
      requesting it on CPU-only hardware raises an error rather than
      silently doing nothing, so build_trainer() resolves this
      explicitly (config.fp16 AND torch.cuda.is_available()) rather
      than passing the config value straight through.
    • eval_every/save_every (in TrainingConfig) are validated but NOT
      YET wired into build_trainer() -- TrainingArguments only
      directly supports "every epoch" (what's hardcoded below) or
      step-based intervals, not "every N epochs" without extra
      steps-per-epoch math.
    • This file NEVER imports anything from backend/ -- training and
      serving are deliberately decoupled (docs/SDP.md Section 5).

REFERENCES
----------
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr
    - https://huggingface.co/docs/transformers/main/en/main_classes/trainer
    - https://huggingface.co/docs/transformers/tasks/object_detection
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.data.augmentations import (
    apply_transforms,
    get_eval_transforms,
    get_train_transforms,
)
from src.data.dataloader import collate_fn
from src.data.dataset import load_split
from src.data.preprocessing import get_image_processor, preprocess_batch
from src.eval.evaluate import (
    compute_detection_metrics,
    format_predictions_for_torchmetrics,
    format_targets_for_torchmetrics,
)
from src.training.config import TrainingConfig, load_config
from src.utils.io import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# This project's class taxonomy. SKU-110K is a single-class
# density/counting dataset -- every product instance shares one
# generic "product" label. If per-SKU/per-brand classes are added
# later, update this mapping AND retrain from scratch.
ID2LABEL = {0: "product"}
LABEL2ID = {"product": 0}


class SyncCheckpointToDriveCallback:
    """After each LOCAL checkpoint save completes, copy it to Drive.

    See this module's "CHECKPOINT STORAGE" note for why this exists:
    Trainer writing directly to a Drive-mounted output_dir silently
    dropped the largest files (model.safetensors, optimizer.pt) on
    every save across an entire run. Training to local disk and then
    copying the already-complete folder to Drive avoids that failure
    mode entirely.

    Only the LATEST synced checkpoint is kept on Drive (the previous
    one is deleted before copying the new one), to respect limited
    Drive storage -- if you need multiple historical checkpoints on
    Drive, increase this class's scope accordingly.
    """

    def __init__(self, drive_dir: str):
        """Store the Drive destination directory.

        Args:
            drive_dir: Path (typically under /content/drive/MyDrive/...)
                to sync the latest local checkpoint to.
        """
        self.drive_dir = Path(drive_dir)

    def on_save(self, args, state, control, **kwargs):
        """Hugging Face Trainer callback hook, fired after each checkpoint save.

        Args:
            args: The TrainingArguments in use (provides output_dir).
            state: The TrainerState (provides global_step, used to
                find the just-saved checkpoint folder's name).
            control: The TrainerControl object (returned unmodified).
            **kwargs: Other values Trainer passes to callbacks, unused.

        Returns:
            The same control object, unmodified -- this callback only
            performs a side effect (copying files), it never changes
            training control flow.
        """
        local_checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not local_checkpoint_dir.exists():
            return control

        if self.drive_dir.exists():
            shutil.rmtree(self.drive_dir)
        shutil.copytree(local_checkpoint_dir, self.drive_dir)
        print(f"Synced checkpoint-{state.global_step} to Drive at {self.drive_dir}")

        return control


def _get_configured_image_processor(config: TrainingConfig) -> Any:
    """Load the image processor with its resize target forced to match config.

    Args:
        config: Resolved TrainingConfig.

    Returns:
        A configured AutoImageProcessor instance, with `.size` set to
        {"height": config.image_size, "width": config.image_size}.
    """
    image_processor = get_image_processor(config.model_checkpoint)
    image_processor.size = {"height": config.image_size, "width": config.image_size}
    return image_processor


def build_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load the pretrained RT-DETR model and its matching image processor.

    Args:
        config: Resolved TrainingConfig, used for
            config.model_checkpoint.

    Returns:
        A (model, image_processor) tuple, ready for fine-tuning.
    """
    from transformers import AutoModelForObjectDetection

    model = AutoModelForObjectDetection.from_pretrained(
        config.model_checkpoint,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # Required: the pretrained checkpoint's classification head was
        # trained on COCO's 80 classes; we're replacing it with our
        # single-class head, so the head's weight shape necessarily
        # differs from the pretrained checkpoint's.
        ignore_mismatched_sizes=True,
    )
    image_processor = _get_configured_image_processor(config)
    return model, image_processor


def _preprocess_and_unbatch(examples: dict, image_processor: Any) -> dict:
    """Adapt preprocess_batch's batch-stacked output for with_transform().

    Args:
        examples: A batch dict (dict of lists), as passed by
            dataset.with_transform().
        image_processor: An AutoImageProcessor instance.

    Returns:
        A dict with "pixel_values" (list of per-example tensors) and
        "labels" (list of per-example label dicts), same length as
        the input batch.
    """
    encoded = preprocess_batch(examples, image_processor)
    return {
        "pixel_values": list(encoded["pixel_values"].unbind(0)),
        "labels": encoded["labels"],
    }


def _augment_and_preprocess(
    examples: dict, transforms: Any, image_processor: Any
) -> dict:
    """Apply augmentation + preprocessing together, lazily, via with_transform.

    apply_transforms() (src/data/augmentations.py) operates on ONE
    example at a time, but with_transform() always calls its callback
    with a BATCHED dict (one list per column) -- this function
    bridges that gap: unbatches, augments each example individually,
    then re-batches for preprocess_batch(). Nothing here is ever
    written to Arrow storage -- it's recomputed fresh on every access,
    which fixes both a prior tensor-to-list Arrow coercion bug and a
    2GB Arrow chunk overflow, and correctly gives every epoch a fresh
    random augmentation.

    Args:
        examples: A batched dict ("image", "image_id", "annotations"
            as parallel lists), as passed by with_transform().
        transforms: An albumentations.Compose pipeline (train or eval).
        image_processor: An AutoImageProcessor instance.

    Returns:
        A dict with "pixel_values" and "labels", batched, matching
        _preprocess_and_unbatch()'s output shape.
    """
    batch_size = len(examples["image"])
    augmented_images = []
    augmented_annotations = []

    for i in range(batch_size):
        single_example = {
            "image": examples["image"][i],
            "image_id": examples["image_id"][i],
            "annotations": examples["annotations"][i],
        }
        result = apply_transforms(single_example, transforms)
        augmented_images.append(result["image"])
        augmented_annotations.append(result["annotations"])

    batch_for_preprocessing = {
        "image": augmented_images,
        "image_id": examples["image_id"],
        "annotations": augmented_annotations,
    }
    return _preprocess_and_unbatch(batch_for_preprocessing, image_processor)


def build_datasets(config: TrainingConfig) -> tuple[Any, Any]:
    """Build the preprocessed, augmented train and validation datasets.

    Both augmentation and preprocessing are applied lazily via
    with_transform() -- never via map() -- so image data is never
    written to the dataset's Arrow storage.

    Args:
        config: Resolved TrainingConfig, used for config.train_manifest,
            config.val_manifest, and config.image_size.

    Returns:
        A (train_dataset, val_dataset) tuple, ready to be passed
        directly to a Hugging Face Trainer via build_trainer().
    """
    image_processor = _get_configured_image_processor(config)

    train_dataset = load_split(config.train_manifest)
    val_dataset = load_split(config.val_manifest)

    train_transforms = get_train_transforms(config.image_size)
    eval_transforms = get_eval_transforms(config.image_size)

    def _train_transform(examples: dict) -> dict:
        return _augment_and_preprocess(examples, train_transforms, image_processor)

    def _eval_transform(examples: dict) -> dict:
        return _augment_and_preprocess(examples, eval_transforms, image_processor)

    train_dataset = train_dataset.with_transform(_train_transform)
    val_dataset = val_dataset.with_transform(_eval_transform)

    return train_dataset, val_dataset


def _build_raw_eval_dataset(config: TrainingConfig) -> Any:
    """Build the val split for standalone metric computation (not for the Trainer).

    Args:
        config: Resolved TrainingConfig.

    Returns:
        A Dataset with "image" (PIL.Image, resized), "image_id", and
        "annotations" (COCO XYWH format) fields.
    """
    eval_transforms = get_eval_transforms(config.image_size)
    dataset = load_split(config.val_manifest)

    def _transform(examples: dict) -> dict:
        batch_size = len(examples["image"])
        images = []
        annotations = []
        for i in range(batch_size):
            single_example = {
                "image": examples["image"][i],
                "image_id": examples["image_id"][i],
                "annotations": examples["annotations"][i],
            }
            result = apply_transforms(single_example, eval_transforms)
            images.append(result["image"])
            annotations.append(result["annotations"])
        return {
            "image": images,
            "image_id": examples["image_id"],
            "annotations": annotations,
        }

    dataset = dataset.with_transform(_transform)
    return dataset


def evaluate_on_dataset(
    model: Any,
    image_processor: Any,
    raw_eval_dataset: Any,
    batch_size: int = 8,
    confidence_threshold: float = 0.5,
) -> dict[str, float]:
    """Run a manual forward-pass evaluation loop and compute real detection metrics.

    Args:
        model: The trained model, in eval mode.
        image_processor: The matching, size-configured image processor.
        raw_eval_dataset: Output of _build_raw_eval_dataset().
        batch_size: Number of images per forward-pass batch.
        confidence_threshold: Minimum confidence for a detection to
            count as a prediction.

    Returns:
        A dict with keys "map", "map_50", "map_50_95", "precision",
        "recall", "f1".
    """
    device = next(model.parameters()).device
    model.eval()

    all_predictions: list[dict] = []
    all_targets: list[dict] = []

    num_examples = len(raw_eval_dataset)
    for start in range(0, num_examples, batch_size):
        end = min(start + batch_size, num_examples)
        batch = raw_eval_dataset[start:end]
        images = batch["image"]
        annotations = batch["annotations"]

        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            raw_outputs = model(**inputs)

        height, width = images[0].size[1], images[0].size[0]  # PIL: (width, height)
        target_sizes = [(height, width)] * len(images)

        batch_predictions = format_predictions_for_torchmetrics(
            raw_outputs,
            image_processor,
            target_sizes,
            confidence_threshold=confidence_threshold,
        )
        batch_targets = format_targets_for_torchmetrics(annotations)

        all_predictions.extend(batch_predictions)
        all_targets.extend(batch_targets)

    return compute_detection_metrics(all_predictions, all_targets)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Optional, experimental Trainer-wired metrics callback. NOT used by default.

    Args:
        eval_pred: An EvalPrediction from the Trainer.

    Returns:
        A dict of metric name to float value.

    Raises:
        NotImplementedError: Always -- intentionally left unimplemented.
            Use evaluate_on_dataset() instead.
    """
    raise NotImplementedError(
        "compute_metrics() is intentionally not implemented -- see this "
        "function's docstring. Use evaluate_on_dataset() instead, which "
        "is what main() actually relies on."
    )


def build_trainer(
    model: Any,
    image_processor: Any,
    train_dataset: Any,
    val_dataset: Any,
    config: TrainingConfig,
) -> Any:
    """Construct a fully configured Hugging Face Trainer.

    Args:
        model: The model returned by build_model().
        image_processor: The image processor returned by build_model().
        train_dataset: The training dataset returned by build_datasets().
        val_dataset: The validation dataset returned by build_datasets().
        config: Resolved TrainingConfig, used to build TrainingArguments.

    Returns:
        A transformers.Trainer instance, ready for .train() to be called.
    """
    from transformers import Trainer, TrainingArguments

    use_fp16 = config.fp16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.scheduler,
        warmup_steps=config.warmup_steps,
        fp16=use_fp16,
        dataloader_num_workers=config.num_workers,
        dataloader_persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
        dataloader_pin_memory=config.pin_memory,
        seed=config.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        logging_steps=config.log_every,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    if config.drive_sync_dir:
        trainer.add_callback(SyncCheckpointToDriveCallback(config.drive_sync_dir))

    return trainer


def _resolve_resume_checkpoint(output_dir: str) -> str | None:
    """Find the last saved checkpoint in output_dir, if any exists.

    Args:
        output_dir: The training output directory (config.output_dir).

    Returns:
        Path to the last checkpoint, or None if output_dir doesn't
        exist yet or contains no checkpoints (i.e. this is a fresh run).
    """
    from transformers.trainer_utils import get_last_checkpoint

    if not os.path.isdir(output_dir):
        return None
    return get_last_checkpoint(output_dir)


def _log_experiment(
    config: TrainingConfig,
    metrics: dict[str, float],
    log_path: str = "experiments/experiment_log.md",
) -> None:
    """Append this run's config and final metrics to the experiment log.

    Args:
        config: The TrainingConfig used for this run.
        metrics: The metrics returned by evaluate_on_dataset().
        log_path: Path to the markdown log file (append-only).

    Returns:
        None. Writes to log_path as a side effect.
    """
    ensure_dir(str(Path(log_path).parent))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"\n## Run: {timestamp}\n",
        f"- model_checkpoint: {config.model_checkpoint}",
        f"- image_size: {config.image_size}",
        f"- batch_size: {config.batch_size}",
        f"- num_epochs: {config.num_epochs}",
        f"- learning_rate: {config.learning_rate}",
        f"- weight_decay: {config.weight_decay}",
        f"- scheduler: {config.scheduler}",
        f"- warmup_steps: {config.warmup_steps}",
        f"- seed: {config.seed}",
        "",
        "**Validation metrics:**",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(
    config_path: str = "configs/training_config.yaml", resume: bool = True
) -> dict[str, float]:
    """Run one full fine-tuning pass end to end, resuming from a checkpoint if found.

    Args:
        config_path: Path to the YAML training configuration file.
        resume: If True (default), automatically resume -- either from
            config.resume_from_checkpoint if explicitly set, or
            otherwise auto-detected as the last checkpoint in
            config.output_dir. Pass False to force a fresh run even if
            checkpoints exist.

    Returns:
        The final validation metrics dict from evaluate_on_dataset().
    """
    from transformers import set_seed

    config = load_config(config_path)
    set_seed(config.seed)
    logger.info("Starting training run with config: %s", config)

    train_dataset, val_dataset = build_datasets(config)
    model, image_processor = build_model(config)
    trainer = build_trainer(model, image_processor, train_dataset, val_dataset, config)

    resume_checkpoint = None
    if resume:
        if config.resume_from_checkpoint:
            resume_checkpoint = config.resume_from_checkpoint
            logger.info(
                "Using explicit resume_from_checkpoint from config: %s",
                resume_checkpoint,
            )
        else:
            resume_checkpoint = _resolve_resume_checkpoint(config.output_dir)

    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
    else:
        logger.info("No checkpoint found -- starting a fresh training run")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model(config.output_dir)
    image_processor.save_pretrained(config.output_dir)

    raw_eval_dataset = _build_raw_eval_dataset(config)
    metrics = evaluate_on_dataset(
        model, image_processor, raw_eval_dataset, batch_size=config.batch_size
    )
    logger.info("Final validation metrics: %s", metrics)

    _log_experiment(config, metrics)

    return metrics


if __name__ == "__main__":
    main()
