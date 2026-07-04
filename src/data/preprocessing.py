"""
src.data.preprocessing

RESPONSIBILITY
--------------
Wraps RT-DETR's `AutoImageProcessor` to convert raw
(image, COCO-format annotations) pairs into the exact pixel_values +
labels tensors the model expects — resizing images, normalizing
pixel values, and converting boxes from absolute XYWH to normalized
CXCYWH. This is the single place that conversion logic lives; it
must be reused identically by BOTH training (src/training/train.py)
and inference (src/inference/inference.py) — a mismatch between how
training and inference preprocess images is one of the most common,
hardest-to-debug sources of silently degraded model performance.

PURPOSE
-------
docs/SDP.md Day 4's explicit deliverable is verifying this
conversion is correct BEFORE trusting any training run — this file
is where that verification target lives.

ARCHITECTURE NOTES
-------------------
    • get_image_processor() is trivial infra (a from_pretrained()
      call) and is implemented for you below, as your reference
      pattern.
    • preprocess_batch() and format_annotations_for_processor() are
      the real logic and are left for you to implement.
    • src/inference/inference.py should IMPORT and reuse
      get_image_processor() (and ideally the same box-formatting
      helpers) from this file rather than duplicating the loading
      logic — this is exactly the training/inference consistency
      concern raised above.

ASCII FLOW DIAGRAM
-------------------
    raw example: {"image": PIL.Image, "annotations": [{"bbox": [x,y,w,h], "category_id": int}, ...]}
            |
            v
    format_annotations_for_processor()   <- YOU implement this
            |  (builds the {"image_id": ..., "annotations": [...]} dict
            |   AutoImageProcessor's `annotations` argument expects)
            v
    preprocess_batch(examples, image_processor)   <- YOU implement this
            |  (calls image_processor(images=..., annotations=..., return_tensors="pt"))
            v
    {"pixel_values": Tensor, "labels": [{"boxes": ..., "class_labels": ...}, ...]}

TODO
----
    - [ ] Implement format_annotations_for_processor(image_id, boxes,
          labels):
          Build the per-image dict AutoImageProcessor expects:
          {"image_id": image_id, "annotations": [{"bbox": [...],
          "category_id": ..., "area": ..., "iscrowd": 0}, ...]}.
          Note `area` is required by the COCO format the processor
          expects — compute it from bbox width * height if your
          manifest doesn't already store it.
    - [ ] Implement preprocess_batch(examples, image_processor):
          1. For each example, call format_annotations_for_processor()
          2. Call image_processor(images=..., annotations=...,
             return_tensors="pt") on the batch
          3. Return the resulting dict of tensors, shaped for the
             Hugging Face Trainer's expectations
    - [ ] AFTER implementing both, write a small verification cell/
          test that re-plots the processed boxes on top of the
          processed (resized) image and visually confirms nothing
          broke — this is Day 4's explicit deliverable in
          docs/SDP.md and is not optional polish.

HINTS
-----
    - `AutoImageProcessor.from_pretrained(checkpoint)` for
      "PekingU/rtdetr_v2_r50vd" returns an RTDetrImageProcessor
      instance — inspect its __call__ signature directly
      (help(image_processor.__call__)) rather than guessing the
      expected annotation dict shape from memory.
    - Batch this with `dataset.map(preprocess_batch, batched=True,
      fn_kwargs={"image_processor": processor})` in
      src/training/train.py's build_datasets(), not by looping
      manually.

COMMON MISTAKES
----------------
    - Forgetting the `area` field in the per-annotation dict — some
      image processors silently misbehave or error without it.
    - Applying augmentations.py's transforms AFTER this
      preprocessing step instead of before — augmentation must
      happen on the raw image/boxes, then THIS module converts the
      already-augmented result into model tensors. Get the order
      backwards and your augmented boxes will be misaligned with
      the (separately) resized image.

BEST PRACTICES
---------------
    - Keep this module's public functions free of any
      training-vs-inference branching — if inference needs slightly
      different behavior (e.g. no augmentation), that's handled by
      simply not calling augmentations.py, not by adding an
      `is_training` flag into this file.

LEARNING NOTES
--------------
get_image_processor() below is intentionally the ONE fully
implemented function in this file — same pattern as
backend/schemas.py's BoundingBox: a trivial wrapper you shouldn't
have to reinvent, next to real logic you should.

REFERENCES
----------
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr#transformers.RTDetrImageProcessor
"""

from typing import Any


def get_image_processor(checkpoint: str) -> Any:
    """Load the AutoImageProcessor matching a given model checkpoint.

    Fully implemented — trivial wrapper, not worth reinventing.

    Args:
        checkpoint: Hugging Face Hub identifier or local path (e.g.
            "PekingU/rtdetr_v2_r50vd").

    Returns:
        A configured AutoImageProcessor instance.
    """
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(checkpoint)


def format_annotations_for_processor(
    image_id: int,
    boxes: list[list[float]],
    labels: list[int],
) -> dict:
    """Build the per-image annotation dict AutoImageProcessor expects.

    Args:
        image_id: Unique integer identifier for this image.
        boxes: List of [x, y, width, height] boxes in absolute pixel
            coordinates (COCO XYWH format).
        labels: List of integer category IDs, one per box, same
            length and order as `boxes`.

    Returns:
        A dict shaped as {"image_id": ..., "annotations": [...]},
        matching the format AutoImageProcessor's `annotations`
        argument expects (including a computed "area" per box).

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("format_annotations_for_processor() is not implemented yet")


def preprocess_batch(examples: dict, image_processor: Any) -> dict:
    """Convert a batch of raw examples into model-ready tensors.

    Args:
        examples: A batch of raw examples (as produced by
            dataset.map(..., batched=True)), each containing at
            least "image" and "annotations" fields.
        image_processor: An AutoImageProcessor instance, as returned
            by get_image_processor().

    Returns:
        A dict with "pixel_values" and "labels" keys, in the exact
        shape the Hugging Face Trainer / RT-DETR model expects.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("preprocess_batch() is not implemented yet")
