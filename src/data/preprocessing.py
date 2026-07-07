"""
src.data.preprocessing

RESPONSIBILITY
--------------
Wraps RT-DETR's `AutoImageProcessor` to convert raw
(image, COCO-format annotations) pairs into the exact pixel_values +
labels tensors the model expects. This is the single place that
conversion logic lives; it must be reused identically by BOTH
training (src/training/train.py) and inference
(src/inference/inference.py).

ARCHITECTURE NOTES
-------------------
    • get_image_processor() is trivial infra, fully implemented.
    • format_annotations_for_processor() and preprocess_batch() are
      the real logic.
    • Annotations reaching this module have already passed through
      src/data/dataset.py's _validate_annotation() (bbox length,
      numeric types, positive width/height, category_id type) — this
      module does NOT re-validate those fields, to avoid duplicating
      that logic (see dataset.py's docstring: it is deliberately
      "STRICT about the OUTPUT schema it returns" so downstream
      modules can stay simple). What THIS module validates is
      specific to its own new inputs: the area computation and
      per-image batch alignment.
    • Images with zero annotations (e.g. an empty shelf photo) are
      valid, supported input — they simply produce an empty
      "annotations" list, a legitimate negative training example.

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
    """Convert one image's annotations into AutoImageProcessor's expected shape.

    Args:
        image_id: Unique integer identifier for this image.
        boxes: List of [x, y, width, height] boxes in absolute pixel
            coordinates (COCO XYWH format).
        labels: List of integer category IDs, one per box, same
            length and order as `boxes`.

    Returns:
        A dict shaped as {"image_id": ..., "annotations": [...]},
        where each annotation includes "bbox", "category_id", "area"
        (width * height — correct for this project's box-only
        annotations; would need to come from polygon area instead if
        segmentation masks are ever added), and "iscrowd" (always 0
        — we have no crowd/group annotations).


    """
    if len(boxes) != len(labels):
        raise ValueError(
            f"boxes and labels must have the same length, got "
            f"{len(boxes)} boxes and {len(labels)} labels for image_id={image_id}."
        )

    annotations = []
    for bbox, category_id in zip(boxes, labels):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"bbox must be [x, y, width, height] (4 numbers), got: {bbox!r} "
                f"for image_id={image_id}."
            )
        if not all(isinstance(v, (int, float)) for v in bbox):
            raise ValueError(
                f"bbox values must all be numeric (int or float), got: {bbox!r} "
                f"for image_id={image_id}."
            )

        x, y, width, height = bbox
        if x < 0 or y < 0:
            raise ValueError(
                f"bbox x/y must be non-negative, got: {bbox!r} for image_id={image_id}."
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"bbox width/height must be positive, got: {bbox!r} "
                f"for image_id={image_id}."
            )

        area = width * height

        annotations.append(
            {
                "bbox": [x, y, width, height],
                "category_id": category_id,
                "area": area,
                "iscrowd": 0,
            }
        )

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def preprocess_batch(
    examples: dict[str, list[Any]], image_processor: Any
) -> dict[str, Any]:
    """Convert a batch of raw examples into model-ready tensors.

    Args:
        examples: A batch dict as produced by
            dataset.map(preprocess_batch, batched=True) — each key
            ("image", "image_id", "annotations") maps to a list with
            one entry per example in the batch, guaranteed by the
            Dataset's schema (see src/data/dataset.py's load_split).
        image_processor: An AutoImageProcessor instance, as returned
            by get_image_processor().

    Returns:
        A dict with "pixel_values" and "labels" keys, in the shape
        the Hugging Face Trainer / RT-DETR model expects.

    """
    images = examples["image"]
    image_ids = examples["image_id"]
    annotation_lists = examples["annotations"]

    if not (len(images) == len(image_ids) == len(annotation_lists)):
        raise ValueError(
            f"Batch fields must all have the same length — got "
            f"{len(images)} images, {len(image_ids)} image_ids, "
            f"{len(annotation_lists)} annotation lists. A mismatch here "
            f"would silently misalign images with the wrong annotations."
        )

    formatted_annotations = []
    for image_id, annotation_list in zip(image_ids, annotation_lists):
        try:
            boxes = [ann["bbox"] for ann in annotation_list]
            labels = [ann["category_id"] for ann in annotation_list]
        except KeyError as exc:
            raise ValueError(
                f"Malformed annotation for image_id={image_id}: missing key {exc}. "
                f"Expected each annotation to already have 'bbox' and 'category_id' "
                f"(guaranteed by src/data/dataset.py's load_split) — this suggests "
                f"preprocess_batch was called on data that bypassed that step."
            ) from exc

        formatted_annotations.append(
            format_annotations_for_processor(
                image_id=image_id,
                boxes=boxes,
                labels=labels,
            )
        )

    encoded = image_processor(
        images=images,
        annotations=formatted_annotations,
        return_tensors="pt",
    )

    return encoded
