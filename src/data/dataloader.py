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

PURPOSE
-------
docs/SDP.md Day 6's explicit deliverable is sanity-checking one full
batch by loading and plotting it — this file is what that batch
comes from. Getting the collate function right here is what prevents
a confusing shape-mismatch error deep inside model.forward() later.

ARCHITECTURE NOTES
-------------------
    • If you use the Hugging Face Trainer (the recommended path per
      docs/SDP.md Section 8), you pass collate_fn to Trainer's
      `data_collator` argument — you do NOT need to manually
      construct a DataLoader yourself for training. build_dataloader()
      here exists for the cases OUTSIDE the Trainer: quick manual
      batch inspection (Day 6), or a custom evaluation loop in
      src/eval/evaluate.py if you choose not to rely solely on
      Trainer's built-in evaluation.
    • pixel_values (fixed-size, per docs/SDP.md's image_size config)
      can be stacked into a single tensor via default collation.
      labels (variable number of boxes per image) CANNOT — they must
      be collected into a plain Python list, one dict per image, not
      stacked into a tensor.

ASCII FLOW DIAGRAM
-------------------
    list[preprocessed example]   (from src/data/preprocessing.py)
            |
            v
    collate_fn(batch)             <- YOU implement this
            |   pixel_values: stack into one tensor
            |   labels: keep as a list of per-image dicts
            v
    {"pixel_values": Tensor[B, C, H, W], "labels": [dict, dict, ...]}

TODO
----
    - [ ] Implement collate_fn(batch):
          1. torch.stack() the "pixel_values" field across the batch
          2. Leave "labels" as a plain list (one entry per example,
             each entry itself a dict with "boxes" and
             "class_labels" — do NOT try to stack this into a
             tensor, the whole point is that it's variable-length)
          3. Return {"pixel_values": ..., "labels": ...}

HINTS
-----
    - This is one of the most commonly mismatched pieces of a
      DETR-family training pipeline — if you hit a shape error
      inside model.forward(), check this function first before
      assuming the bug is in preprocessing.py.

COMMON MISTAKES
----------------
    - Trying to torch.stack() the labels field — this will fail (or
      worse, silently produce garbage if box counts happen to match
      across a batch by coincidence) because it is fundamentally
      variable-length data.

BEST PRACTICES
---------------
    - Write a tiny unit test that constructs 2-3 fake preprocessed
      examples with DIFFERENT numbers of boxes each and asserts
      collate_fn() doesn't raise and produces the expected shape —
      this is a fast, valuable test to have in
      tests/test_preprocessing.py.

LEARNING NOTES
--------------
build_dataloader() below is fully implemented as a thin, generic
wrapper — the real learning value in this file is entirely in
getting collate_fn() correct.

REFERENCES
----------
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    - https://huggingface.co/docs/transformers/main/en/main_classes/data_collator
"""

from typing import Any


def collate_fn(batch: list[dict]) -> dict:
    """Collate a list of preprocessed examples into one training batch.

    Args:
        batch: A list of preprocessed example dicts, each with
            "pixel_values" (a fixed-shape tensor) and "labels" (a
            dict describing that image's boxes/classes).

    Returns:
        A single dict with "pixel_values" stacked into one batched
        tensor and "labels" left as a plain list of per-image dicts.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("collate_fn() is not implemented yet")


def build_dataloader(dataset: Any, batch_size: int, shuffle: bool) -> Any:
    """Build a standalone PyTorch DataLoader for manual/eval use.

    Fully implemented — thin wrapper, not worth reinventing. NOTE:
    the Hugging Face Trainer does not use this function directly; it
    builds its own internal DataLoader using the collate_fn you pass
    to TrainingArguments/Trainer instead. Use this function for
    manual batch inspection (docs/SDP.md Day 6) or a custom eval
    loop only.

    Args:
        dataset: A preprocessed dataset (torch-compatible).
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle example order each epoch.

    Returns:
        A configured torch.utils.data.DataLoader using collate_fn
        from this module.
    """
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
