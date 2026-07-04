"""
src.data.augmentations

RESPONSIBILITY
--------------
Defines the Albumentations-based augmentation pipelines used for the
training split (get_train_transforms) and the deterministic,
augmentation-free pipeline used for validation/test/inference
(get_eval_transforms), plus the glue function that applies either
pipeline to one dataset example while keeping bounding boxes in
sync with the transformed image.

ARCHITECTURE NOTES
-------------------
    • get_eval_transforms() only resizes — no geometry-changing
      randomness, so boxes never need to be dropped or clipped.
    • get_train_transforms() deliberately excludes A.VerticalFlip —
      a shelf camera never sees an upside-down shelf in production
      (docs/SDP.md Day 5). It also deliberately excludes any
      CROPPING transform (e.g. RandomCrop) for now — cropping can
      cut a box partially or fully out of frame, which would require
      min_visibility/min_area handling in BboxParams that the
      current simple recipe (resize/flip/brightness) doesn't need,
      since none of those three transforms can push a box out of
      frame. If you add a crop-based transform later, revisit this.
    • apply_transforms() operates on ONE example at a time (not
      batched) — wire it in via dataset.map(apply_transforms,
      fn_kwargs={"transforms": ...}) WITHOUT batched=True, since
      Albumentations' Compose call signature expects a single
      image/bboxes/labels triple, not a batch of them.

REFERENCES
----------
    - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
"""

from typing import Any

import numpy as np
from PIL import Image


def get_eval_transforms(image_size: int) -> Any:
    """Build the deterministic, augmentation-free transform pipeline.

    Fully implemented — used for validation, test, and (via
    src/inference/inference.py) real inference requests, where
    randomized augmentation would be inappropriate.

    Args:
        image_size: Target square image size in pixels.

    Returns:
        An albumentations.Compose pipeline that only resizes the
        image, with matching bounding-box handling.
    """
    import albumentations as A

    return A.Compose(
        [A.Resize(image_size, image_size)],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )


def get_train_transforms(image_size: int) -> Any:
    """Build the randomized training-time augmentation pipeline.

    Args:
        image_size: Target square image size in pixels.

    Returns:
        An albumentations.Compose pipeline including randomized
        augmentations appropriate for shelf imagery. Deliberately
        excludes vertical flips (shelves are never viewed upside
        down) and any cropping transform (see this file's module
        docstring). Transforms bounding boxes in sync with the image
        via bbox_params.
    """
    import albumentations as A

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )


def apply_transforms(example: dict, transforms: Any) -> dict:
    """Apply an Albumentations pipeline to a single dataset example.

    Args:
        example: A raw dataset example with "image" (a PIL.Image,
            as lazily decoded by src/data/dataset.py's Image()-cast
            column), "image_id", and "annotations" (a list of
            {"bbox": [x, y, w, h], "category_id": int} dicts, in the
            schema produced by src/data/dataset.py's load_split()).
        transforms: An albumentations.Compose pipeline, as returned
            by get_train_transforms() or get_eval_transforms().

    Returns:
        A new example dict with the same keys as the input, but with
        "image" replaced by the transformed PIL.Image and
        "annotations" replaced by boxes remapped to match the
        transformed image. Examples with zero annotations are
        handled correctly (an empty list in, an empty list out) —
        a fully-stocked or fully-empty shelf photo is a legitimate
        input either way.

    Raises:
        ValueError: If any annotation is missing "bbox" or
            "category_id" — this should never happen for data that
            passed through src/data/dataset.py's load_split(), so a
            failure here indicates apply_transforms was called on
            data that bypassed that validation step.
    """
    # ------------------------------------------------------------------
    # Step 1: Albumentations works on numpy arrays, not PIL Images.
    # Force RGB explicitly (same convention as src/utils/io.py's
    # load_image()) — some source images are grayscale or RGBA, and
    # silently mixing image modes through the pipeline is a classic
    # subtle bug.
    # ------------------------------------------------------------------
    image = example["image"]
    image_np = np.array(image.convert("RGB"))

    # ------------------------------------------------------------------
    # Step 2: Split this project's {"bbox": ..., "category_id": ...}
    # annotation dicts into the two SEPARATE parallel lists
    # Albumentations requires: `bboxes` and `labels` (tied together
    # only by matching list position, per bbox_params(label_fields=
    # ["labels"]) in get_train_transforms/get_eval_transforms above).
    # ------------------------------------------------------------------
    try:
        bboxes = [ann["bbox"] for ann in example["annotations"]]
        labels = [ann["category_id"] for ann in example["annotations"]]
    except KeyError as exc:
        raise ValueError(
            f"Malformed annotation for image_id={example.get('image_id')}: "
            f"missing key {exc}. Expected annotations already validated by "
            f"src/data/dataset.py's load_split()."
        ) from exc

    # ------------------------------------------------------------------
    # Step 3: Call the pipeline. Albumentations moves the boxes
    # automatically to match whatever happened to the image (resize,
    # flip, etc.) — this is the one line doing the actual "work."
    # ------------------------------------------------------------------
    transformed = transforms(image=image_np, bboxes=bboxes, labels=labels)

    # ------------------------------------------------------------------
    # Step 4: Reassemble the result back into this project's schema.
    # Convert the transformed numpy array back to a PIL.Image so the
    # "image" field's TYPE stays consistent with what
    # src/data/dataset.py produces and src/data/preprocessing.py
    # expects — nothing downstream should need to know augmentation
    # happened at all.
    # ------------------------------------------------------------------
    transformed_image = Image.fromarray(transformed["image"])
    transformed_annotations = [
        {"bbox": list(bbox), "category_id": int(label)}
        for bbox, label in zip(transformed["bboxes"], transformed["labels"])
    ]

    result = dict(example)
    result["image"] = transformed_image
    result["annotations"] = transformed_annotations
    return result
