"""
src.data.dataset

RESPONSIBILITY
--------------
Loads the train/val/test split manifests (data/splits/*.json,
produced once during docs/SDP.md Day 6) into Hugging Face
`datasets.Dataset` objects with COCO-detection-shaped fields
(image, image_id, annotations: [{bbox, category_id}, ...]) — the
exact shape `AutoImageProcessor` expects downstream in
preprocessing.py.

PURPOSE
-------
This file is the boundary between "files on disk" and "an object
the rest of the pipeline can work with." Everything after this file
(preprocessing, augmentation, batching) should never need to know
whether the underlying data came from SKU-110K's Roboflow export, a
Hugging Face Hub mirror, or your own supplementary images — it only
ever sees a datasets.Dataset with a consistent schema.

ARCHITECTURE NOTES
-------------------
    • This file assumes data/splits/{train,val,test}.json already
      exist (docs/SDP.md Day 6 deliverable) — it does NOT scrape or
      download SKU-110K itself.
    • load_split() must be format-agnostic about ANNOTATION SOURCE
      but STRICT about the OUTPUT schema it returns.
    • Image paths stored in the manifest are resolved against
      PROJECT_ROOT (this file's location, walked up to the project
      root) rather than the current working directory — this is
      what makes the dataset load correctly whether it's called
      from the repo root, from inside notebooks/, or from Colab.

REFERENCES
----------
    - https://huggingface.co/docs/datasets/en/about_dataset_load
    - https://huggingface.co/docs/datasets/en/image_load
"""

import logging
from pathlib import Path

from datasets import Dataset, Image

from src.utils.io import load_json
from src.utils.logger import get_logger

# This file lives at <project_root>/src/data/dataset.py, so walking
# up two parents from this file's directory (src/data -> src ->
# project_root) gives a cwd-independent anchor for resolving any
# relative image paths stored in a manifest.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = get_logger(__name__, level=logging.INFO)


def _resolve_image_path(image_path: str) -> str:
    """Resolve a manifest's image_path against the project root.

    Args:
        image_path: A path as stored in the manifest — may be
            relative (resolved against PROJECT_ROOT) or already
            absolute (returned unchanged).

    Returns:
        An absolute path string, safe to use regardless of the
        caller's current working directory.
    """
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _validate_annotation(annotation: dict) -> dict:
    """Validate and normalize a single annotation dict.

    Args:
        annotation: A raw annotation dict from the manifest, expected
            to contain "bbox" (list of 4 numbers, COCO XYWH format)
            and "category_id" (int).

    Returns:
        A cleaned {"bbox": [...], "category_id": int} dict.

    Raises:
        ValueError: If "bbox" or "category_id" is missing, if bbox
            is not a 4-element list/tuple of numbers, or if
            category_id is not an int.
    """
    if "bbox" not in annotation:
        raise ValueError(f"Annotation missing required 'bbox' key: {annotation}")
    if "category_id" not in annotation:
        raise ValueError(f"Annotation missing required 'category_id' key: {annotation}")

    bbox = annotation["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must be a list/tuple of 4 numbers, got: {bbox!r}")
    if not all(isinstance(v, (int, float)) for v in bbox):
        raise ValueError(f"bbox values must be numeric, got: {bbox!r}")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"bbox width/height must be positive, got: {bbox!r}")

    category_id = annotation["category_id"]
    if not isinstance(category_id, int):
        raise ValueError(f"category_id must be an int, got: {category_id!r}")

    return {"bbox": list(bbox), "category_id": category_id}


def load_split(manifest_path: str) -> Dataset:
    """Load one dataset split from a JSON manifest into a Dataset.

    Args:
        manifest_path: Path to a JSON manifest file (e.g.
            "data/splits/train.json"), where each record has an
            "image_path", "image_id", and a list of "annotations".

    Returns:
        A datasets.Dataset with "image" (lazily-loaded PIL image),
        "image_id", and "annotations" fields, ready for
        src/data/preprocessing.py.

    Raises:
        FileNotFoundError: If manifest_path does not exist (raised
            by load_json).
        ValueError: If the manifest is empty, is not a list of
            records, or contains a record/annotation missing a
            required field or with an invalid value.
    """
    logger.info("Loading manifest: %s", manifest_path)
    records = load_json(manifest_path)

    if not isinstance(records, list):
        raise ValueError(f"Manifest must contain a list of records: {manifest_path}")
    if len(records) == 0:
        raise ValueError(f"Manifest is empty, nothing to load: {manifest_path}")

    cleaned_records = []
    for record in records:
        if "image_path" not in record:
            raise ValueError(f"Record missing required 'image_path' key: {record}")
        if "image_id" not in record:
            raise ValueError(f"Record missing required 'image_id' key: {record}")
        if "annotations" not in record:
            raise ValueError(f"Record missing required 'annotations' key: {record}")

        annotations = record["annotations"]
        if not isinstance(annotations, list):
            raise ValueError(f"'annotations' must be a list: {record}")

        cleaned_annotations = [_validate_annotation(a) for a in annotations]

        cleaned_records.append(
            {
                "image": _resolve_image_path(record["image_path"]),
                "image_id": record["image_id"],
                "annotations": cleaned_annotations,
            }
        )

    dataset = Dataset.from_list(cleaned_records)
    # Cast "image" to lazily-decoded images — avoids loading every
    # image into memory up front, which matters at SKU-110K's scale.

    dataset = dataset.cast_column(
        "image", Image()
    )  # .cast_column Convert one data type into another data type. it

    logger.info("Loaded %d records from %s", len(dataset), manifest_path)
    return dataset


def build_datasets(train_manifest: str, val_manifest: str) -> tuple[Dataset, Dataset]:
    """Load both the training and validation splits.

    Args:
        train_manifest: Path to the training split manifest.
        val_manifest: Path to the validation split manifest.

    Returns:
        A (train_dataset, val_dataset) tuple of raw (not yet
        preprocessed or augmented) datasets.Dataset objects.

    Raises:
        FileNotFoundError: If either manifest path does not exist.
        ValueError: If either manifest is malformed (see load_split).
    """
    train_dataset = load_split(train_manifest)
    val_dataset = load_split(val_manifest)
    return train_dataset, val_dataset
