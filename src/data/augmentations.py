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

PURPOSE
-------
Augmentation must match deployment reality (docs/SDP.md Day 5) — a
shelf camera never sees an upside-down shelf, so vertical flips are
deliberately excluded below, even though many generic augmentation
tutorials include them by default. Blindly copying a generic
augmentation recipe without this kind of domain judgment is exactly
the "tutorial project" energy this whole build is trying to avoid.

ARCHITECTURE NOTES
-------------------
    • get_eval_transforms() is implemented for you — it's
      deterministic (resize only) and there's no meaningful design
      decision left to make once you've decided on image_size.
    • get_train_transforms() is NOT implemented — choosing which
      augmentations to include (and which to deliberately exclude,
      like vertical flips) is a real design decision, not
      boilerplate.
    • apply_transforms() is the bridge between an Albumentations
      Compose object and a single Hugging Face dataset example — it
      must keep image and bounding boxes transformed together
      (Albumentations handles this natively via its
      bbox_params argument, but wiring it into a dataset.map()-
      compatible function is your job).

ASCII FLOW DIAGRAM
-------------------
    dataset example (image, boxes, labels)
            |
            v
    apply_transforms(example, transforms)   <- YOU implement this
            |   (transforms is either get_train_transforms() output,
            |    applied ONLY to the train split, or
            |    get_eval_transforms() output, applied to val/test)
            v
    transformed example (image, boxes, labels — boxes remapped to
    match the transformed image)
            |
            v
    (fed into src/data/preprocessing.py's preprocess_batch())

TODO
----
    - [ ] Implement get_train_transforms(image_size):
          Recommended starting recipe (confirm/adjust after looking
          at your own EDA from Day 3):
              - A.Resize(image_size, image_size)
              - A.HorizontalFlip(p=0.5)
              - A.RandomBrightnessContrast(p=0.3)
              - A.Compose(..., bbox_params=A.BboxParams(format="coco", label_fields=["labels"]))
          Deliberately EXCLUDE A.VerticalFlip — shelves are never
          viewed upside-down in production (docs/SDP.md Day 5).
    - [ ] Implement apply_transforms(example, transforms):
          1. Extract image (as a numpy array) and boxes/labels from
             the example
          2. Call transforms(image=..., bboxes=..., labels=...)
          3. Reassemble the transformed image + boxes back into the
             example's expected shape

HINTS
-----
    - Albumentations expects bboxes as a list of tuples/lists, with
      labels passed as a SEPARATE list_field (via
      label_fields=["labels"] in BboxParams) — mixing labels
      directly into each bbox tuple is a common early mistake.
    - Test apply_transforms() on a single example first and
      visually plot the result (reuse src/utils/visualization.py
      once implemented) before wiring it into a full dataset.map()
      call.

COMMON MISTAKES
----------------
    - Applying get_train_transforms() to the validation/test split —
      this contaminates your evaluation metrics with augmentation
      noise and makes results non-reproducible run to run.
    - Forgetting bbox_params entirely, which silently leaves boxes
      untransformed while the image itself IS transformed —
      producing misaligned boxes that are hard to notice without
      explicitly visualizing them.

BEST PRACTICES
---------------
    - Keep the augmentation recipe itself simple at first (2-3
      transforms) and only add more if error analysis (Day 10) shows
      a specific failure mode augmentation could plausibly address —
      resist "augmentation kitchen-sink" syndrome.

LEARNING NOTES
--------------
get_eval_transforms() below shows the minimal, deterministic-only
pattern; get_train_transforms() should follow the same
Albumentations Compose + BboxParams structure, just with additional
randomized transforms added.

REFERENCES
----------
    - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
"""

from typing import Any


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
        augmentations appropriate for shelf imagery. Must NOT
        include vertical flips (see this file's module docstring for
        why) and must transform bounding boxes in sync with the
        image.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("get_train_transforms() is not implemented yet")


def apply_transforms(example: dict, transforms: Any) -> dict:
    """Apply an Albumentations pipeline to a single dataset example.

    Args:
        example: A raw dataset example containing at least an image
            and its bounding boxes/labels.
        transforms: An albumentations.Compose pipeline, as returned
            by get_train_transforms() or get_eval_transforms().

    Returns:
        A new example dict with the image and boxes/labels
        consistently transformed together.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("apply_transforms() is not implemented yet")
