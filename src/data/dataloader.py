"""
src.data.dataloader

RESPONSIBILITY
--------------
Provides the custom `collate_fn` needed to batch RT-DETR examples
(which have a variable number of boxes per image — a plain
torch default_collate cannot handle this) and a thin
build_dataloader() convenience wrapper around torch.utils.data.DataLoader
for any standalone (non-Trainer) usage, such as quick manual sanity
checks or a custom eval loop.

ARCHITECTURE NOTES
-------------------
    • If you use the Hugging Face Trainer (the recommended path per
      docs/SDP.md Section 8), you pass collate_fn to Trainer's
      `data_collator` argument — you do NOT need to manually
      construct a DataLoader yourself for training. build_dataloader()
      exists for cases OUTSIDE the Trainer only.
    • pixel_values (fixed-size, per configs/training_config.yaml's
      image_size) is stacked into a single batched tensor.
      labels (variable number of boxes per image) is deliberately
      LEFT as a plain Python list, one dict per image — never
      stacked, never padded. This is the one thing this file exists
      to get right.

REFERENCES
----------
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    - https://huggingface.co/docs/transformers/main/en/main_classes/data_collator
"""

from typing import Any

import torch


def collate_fn(batch: list[dict]) -> dict:
    """Collate a list of preprocessed examples into one training batch.

    Args:
        batch: A list of preprocessed example dicts, each with
            "pixel_values" (a fixed-shape [C, H, W] tensor, produced
            by src/data/preprocessing.py's preprocess_batch via
            AutoImageProcessor) and "labels" (a dict with "boxes",
            "class_labels", and related fields for that one image).

    Returns:
        A single dict with "pixel_values" stacked into one
        [batch_size, C, H, W] tensor, and "labels" left as a plain
        list of per-image dicts, in the same order as the input batch.

    Raises:
        ValueError: If `batch` is empty, or if the examples'
            "pixel_values" tensors don't all share the same shape
            (which would indicate a bug upstream in preprocessing —
            every image should already be resized to a consistent
            size before reaching this function).
    """
    if len(batch) == 0:
        raise ValueError("collate_fn() received an empty batch — nothing to collate.")

    pixel_values = [torch.as_tensor(item["pixel_values"]) for item in batch]

    first_shape = pixel_values[0].shape
    for i, tensor in enumerate(pixel_values):
        if tensor.shape != first_shape:
            raise ValueError(
                f"Inconsistent pixel_values shapes in batch: example 0 has shape "
                f"{first_shape}, example {i} has shape {tensor.shape}. All images "
                f"should already be resized to the same size by "
                f"src/data/preprocessing.py — this indicates a bug upstream, not here."
            )

    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": [item["labels"] for item in batch],
    }


def build_dataloader(dataset: Any, batch_size: int, shuffle: bool) -> Any:
    """Build a standalone PyTorch DataLoader for manual/eval use.

    Fully implemented — thin wrapper, not worth reinventing. NOTE:
    the Hugging Face Trainer does not use this function directly; it
    builds its own internal DataLoader using the collate_fn you pass
    to TrainingArguments/Trainer instead. Use this function for
    manual batch inspection or a custom eval loop only.

    Args:
        dataset: A preprocessed dataset (torch-compatible).
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle example order each epoch.

    Returns:
        A configured torch.utils.data.DataLoader using collate_fn
        from this module.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
