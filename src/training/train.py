"""
src.training.train

RESPONSIBILITY
--------------
Orchestrates one full RT-DETR fine-tuning run: build the dataset,
build the model, configure the Hugging Face Trainer, run training,
resume automatically from the last checkpoint if one exists, and run
a standalone, verified evaluation pass at the end.

VERIFIED (confirmed against a real run, see conversation history for
the actual smoke-test output that proved each of these)
------------------------------------------------------------------
    1. _get_configured_image_processor()'s `.size` override -- confirmed
       working: assigning a plain dict to `.size` correctly changes
       the processor's actual resize behavior (verified: a real
       RTDetrImageProcessor resized a test image to the overridden
       size, not the default).
    2. compute_metrics()'s assumed EvalPrediction shape is NOT relied
       on by main(); main() uses evaluate_on_dataset() instead, which
       only depends on functions already unit-tested in
       src/eval/evaluate.py, and has been confirmed working end-to-end
       via a real smoke-test training run.

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
      forward pass deep inside the model (a real bug hit and fixed
      during development: 'list' object has no attribute 'device').
      with_transform() instead applies the function fresh, on every
      single access, so the tensors it returns stay real tensors.
      This matches Hugging Face's own official object detection
      guide, which does the same for exactly this reason.
    • Two different "val dataset" representations exist on purpose:
      the Trainer-ready one (fully processor-encoded, used only for
      Trainer's built-in eval_loss tracking) and a separate raw
      (augmented-but-not-processor-encoded) one built by
      _build_raw_eval_dataset(), used for the real, verified
      mAP/precision/recall/F1 numbers via evaluate_on_dataset().
      Running the forward pass twice is a deliberate, acceptable
      tradeoff for this project's dataset scale -- see
      compute_metrics()'s docstring for why we don't trust Trainer's
      own compute_metrics wiring.
    • fp16 is only meaningful (and only valid) on a CUDA GPU --
      requesting it on CPU-only hardware raises an error rather than
      silently doing nothing, so build_trainer() resolves this
      explicitly (config.fp16 AND torch.cuda.is_available()) rather
      than passing the config value straight through.
    • eval_every/save_every (in TrainingConfig) are validated but NOT
      YET wired into build_trainer() -- TrainingArguments only
      directly supports "every epoch" (what's hardcoded below) or
      step-based intervals, not "every N epochs" without extra
      steps-per-epoch math. Reserved for a future extension if a
      longer training run ever needs less-frequent evaluation.
    • This file NEVER imports anything from backend/ -- training and
      serving are deliberately decoupled (docs/SDP.md Section 5).

ASCII FLOW DIAGRAM
-------------------
    load_config("configs/training_config.yaml")
            |
            v
    build_datasets(config)      -> train_dataset, val_dataset (Trainer-ready)
            |
            v
    build_model(config)          -> model, image_processor
            |
            v
    build_trainer(...)           -> Trainer (remove_unused_columns=False)
            |
            v
    resolve checkpoint (explicit config override, else auto-detect last)
            |
            v
    trainer.train(resume_from_checkpoint=...)
            |
            v
    save model + image_processor to config.output_dir
            |
            v
    _build_raw_eval_dataset(config) -> evaluate_on_dataset(...) -> real metrics
            |
            v
    _log_experiment(config, metrics) -> experiments/experiment_log.md

REFERENCES
----------
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr
    - https://huggingface.co/docs/transformers/main/en/main_classes/trainer
    - https://huggingface.co/docs/transformers/tasks/object_detection
      (official guide confirming the with_transform() pattern above)
"""

import os
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
# later (docs/SDP.md Section 15, Future Improvements), update this
# mapping AND retrain from scratch, since the classification head's
# size depends on it.
ID2LABEL = {0: "product"}
LABEL2ID = {"product": 0}


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
    """Adapt preprocess_batch's batch-stacked output for dataset.with_transform().

    preprocess_batch() (src/data/preprocessing.py) returns
    {"pixel_values": Tensor[B, C, H, W], "labels": [dict, ...]} -- a
    single stacked tensor for the whole batch. datasets' with_transform
    (like map(batched=True)) instead expects a dict of LISTS, one entry
    per row. This function does that unbinding -- it is glue code
    specific to how this project wires src/data/preprocessing.py into
    a Hugging Face Dataset, not duplicate logic.

    Args:
        examples: A batch dict, as passed by dataset.with_transform().
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
    with a BATCHED dict (one list per column) -- this function bridges
    that gap: unbatches, augments each example individually, then
    re-batches for preprocess_batch(). Critically, nothing here ever
    gets written to Arrow storage -- it's recomputed fresh on every
    access, which is both what fixes the 2GB Arrow chunk overflow
    (previously hit when build_datasets() used .map() to permanently
    write every augmented image to disk-backed storage) and what
    correctly gives every epoch a fresh random augmentation, rather
    than reusing one cached augmented version forever.

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
    written to the dataset's Arrow storage (avoiding both the 2GB
    Arrow chunk limit and the earlier tensor-to-list Arrow round-trip
    bug). Each access re-augments and re-preprocesses fresh, which is
    also the correct behavior for randomized training-time augmentation.

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

    Uses with_transform() (not map()) for the same reason as
    build_datasets() -- avoids writing augmented image data to Arrow
    storage. Stops after augmentation only -- boxes stay in COCO XYWH
    format, matching what evaluate_on_dataset() expects.

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

    This is the reliable evaluation path this project actually uses
    (see compute_metrics()'s docstring for why its Trainer-wired
    alternative is not trusted by default). Reuses src/eval/evaluate.py's
    already-tested functions directly -- no Trainer internals involved.

    Args:
        model: The trained model, in eval mode.
        image_processor: The matching, size-configured image processor.
        raw_eval_dataset: Output of _build_raw_eval_dataset() -- NOT
            build_datasets()'s Trainer-ready val_dataset.
        batch_size: Number of images per forward-pass batch.
        confidence_threshold: Minimum confidence for a detection to
            count as a prediction.

    Returns:
        A dict with keys "map", "map_50", "map_50_95", "precision",
        "recall", "f1" (see src/eval/evaluate.py's
        compute_detection_metrics for definitions).
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

        # Images here were already resized to config.image_size by
        # get_eval_transforms() (via _build_raw_eval_dataset), and the
        # image_processor's own resize was forced to match that same
        # size (_get_configured_image_processor) -- so this is the
        # correct, single, consistent frame for both predictions and
        # targets.
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

    Wiring torchmetrics-based detection metrics directly into the
    Hugging Face Trainer's compute_metrics is known to be fragile
    across transformers versions, because the Trainer's default
    evaluation loop concatenates predictions across batches -- which
    does not work cleanly for object detection, where each image has
    a different number of boxes. This function assumes
    TrainingArguments(eval_do_concat_batches=False) is set (NOT done
    by build_trainer() by default), which changes eval_pred.predictions
    into a list of per-batch raw outputs rather than one concatenated
    tensor. This is provided for experimentation only -- main()'s
    real, relied-upon metrics come from evaluate_on_dataset() instead,
    which has no dependency on this function or on Trainer internals,
    and has been confirmed working via a real end-to-end run.

    Args:
        eval_pred: An EvalPrediction from the Trainer, assumed (see
            above) to hold per-batch raw outputs, not concatenated ones.

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

    # fp16 is only meaningful (and only valid) on a CUDA GPU -- requesting
    # it on CPU-only hardware (e.g. local smoke tests) raises an error
    # rather than silently no-op'ing, so this must be resolved
    # explicitly here, not passed straight through from config.
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
        # Keep only the 3 most recent checkpoints -- Colab disk is
        # limited, and we don't need every single epoch's checkpoint
        # once training has moved past it.
        save_total_limit=3,
        load_best_model_at_end=True,
        # eval_loss is used (not a custom detection metric) precisely
        # because it works reliably regardless of Trainer's known
        # fragility with variable-length detection outputs -- see
        # compute_metrics()'s docstring.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Required for object detection models: Trainer's default
        # column-pruning inspects the model's forward signature and
        # will incorrectly strip pixel_values/labels if this is left
        # at its True default.
        remove_unused_columns=False,
        logging_steps=config.log_every,
        report_to=[],  # avoid requiring wandb/etc. unless explicitly configured
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        # compute_metrics intentionally NOT wired here -- see that
        # function's docstring for why. Real metrics come from
        # evaluate_on_dataset(), called separately in main().
    )
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
