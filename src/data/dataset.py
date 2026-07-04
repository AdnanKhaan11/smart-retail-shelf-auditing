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
      download SKU-110K itself. Manifest creation from the raw
      SKU-110K COCO JSON is a one-off task, best done in a notebook
      or a small standalone script, not as part of the reusable
      pipeline this package exposes.
    • load_split() must be format-agnostic about ANNOTATION SOURCE
      but STRICT about the OUTPUT schema it returns — this is what
      lets preprocessing.py stay simple.

ASCII FLOW DIAGRAM
-------------------
    data/splits/train.json (manifest: list of {image_path, annotations})
            |
            v
    load_split(manifest_path)     <- YOU implement this
            |
            v
    datasets.Dataset  (fields: image, image_id, annotations)
            |
            v
    (consumed by src/data/preprocessing.py + augmentations.py,
     orchestrated together in src/training/train.py's build_datasets())

TODO
----
    - [ ] Implement load_split(manifest_path):
          1. Read the JSON manifest (list of records, each with at
             least an image path and a list of box annotations)
          2. Build a datasets.Dataset from it (datasets.Dataset.from_list
             or datasets.Dataset.from_dict, whichever fits your
             manifest's shape better)
          3. Ensure each record's `annotations` field is a list of
             {"bbox": [x, y, w, h], "category_id": int} dicts —
             AutoImageProcessor expects COCO XYWH absolute
             coordinates at this stage (preprocessing.py handles the
             conversion to normalized CXCYWH later — don't do it
             here)
    - [ ] Implement build_datasets(train_manifest, val_manifest):
          A thin convenience wrapper calling load_split() twice.
          Note: src/training/train.py ALSO has a function named
          build_datasets() — that one is the higher-level
          orchestrator that calls THIS function plus
          preprocessing.py and augmentations.py together. Keep the
          naming distinction clear in your own head as you implement
          both.

HINTS
-----
    - If your manifest stores relative image paths, resolve them
      against the project root consistently — a common bug is a
      manifest that works when run from the repo root but breaks
      when run from inside notebooks/.
    - datasets.Dataset can lazily load images if you store paths and
      cast the column with datasets.Image() — this avoids loading
      all images into memory at once for a large dataset.

COMMON MISTAKES
----------------
    - Loading images eagerly into memory during load_split() instead
      of lazily — fine for a small custom dataset, but will not
      scale if you later train on more of SKU-110K's 11,762 images.
    - Silently accepting a manifest with malformed/missing bbox
      fields — validate and raise a clear error instead; a bad
      annotation here fails loudly here, not confusingly three
      layers deeper inside AutoImageProcessor.

BEST PRACTICES
---------------
    - Keep load_split() pure — same input path always produces an
      equivalent Dataset, no hidden randomness (that belongs in
      augmentations.py, applied later, only to the training split).

LEARNING NOTES
--------------
This is a good file to write a focused unit test for
(tests/test_preprocessing.py, or a new test_dataset.py if you want
one) — feed it a tiny 2-3-record fake manifest and assert the
returned Dataset has the expected schema and length.

REFERENCES
----------
    - https://huggingface.co/docs/datasets/en/about_dataset_load
    - https://huggingface.co/docs/datasets/en/image_load
"""

from typing import Any


def load_split(manifest_path: str) -> Any:
    """Load one split manifest into a Hugging Face Dataset.

    Args:
        manifest_path: Path to a JSON manifest file (e.g.
            "data/splits/train.json"), where each record describes
            one image and its COCO-format box annotations.

    Returns:
        A datasets.Dataset with (at minimum) `image`, `image_id`,
        and `annotations` fields, ready for preprocessing.py.

    Raises:
        NotImplementedError: Always, until implemented.
        FileNotFoundError: Should be raised (once implemented) if
            manifest_path does not exist.
    """
    raise NotImplementedError("load_split() is not implemented yet")


def build_datasets(train_manifest: str, val_manifest: str) -> tuple[Any, Any]:
    """Load both the training and validation splits.

    Args:
        train_manifest: Path to the training split manifest.
        val_manifest: Path to the validation split manifest.

    Returns:
        A (train_dataset, val_dataset) tuple of raw (not yet
        preprocessed or augmented) datasets.Dataset objects.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("build_datasets() is not implemented yet")
