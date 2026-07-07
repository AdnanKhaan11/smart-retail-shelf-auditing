"""
scripts/build_manifests.py

RESPONSIBILITY
--------------
One-off, run-once script: converts a COCO-format SKU-110K export
(images/ folder + a COCO annotations JSON, exactly the layout
Roboflow's "COCO" export format produces) into this project's own
manifest schema at data/splits/{train,val,test}.json -- the schema
src/data/dataset.py's load_split() expects.

This is NOT part of the reusable src/ pipeline on purpose (see
src/data/dataset.py's Architecture Notes: manifest creation is "a
one-off task, best done in a notebook or a small standalone script,
not as part of the reusable pipeline"). Run this ONCE per dataset
export, on Colab, where the full ~60-70GB raw dataset actually lives.

WHY THIS SCRIPT EXISTS
------------------------
data/splits/{train,val,test}.json were created as empty placeholder
files by folder_structure.py (Phase 1) -- nothing in this project so
far has ever populated them. This script is that missing step.

EXPECTED INPUT LAYOUT (Roboflow COCO export, one folder per split)
-----------------------------------------------------------------------
    <dataset_root>/
      train/
        _annotations.coco.json
        <image files>
      valid/
        _annotations.coco.json
        <image files>
      test/
        _annotations.coco.json
        <image files>

COCO annotation format assumed (standard, per Roboflow/COCO spec):
    {
      "images": [{"id": int, "file_name": str, ...}, ...],
      "annotations": [{"image_id": int, "bbox": [x,y,w,h],
                        "category_id": int, ...}, ...],
      "categories": [{"id": int, "name": str}, ...]
    }

OUTPUT SCHEMA (matches src/data/dataset.py's load_split() exactly)
-----------------------------------------------------------------------
    [
      {
        "image_path": "data/raw/train/image_0001.jpg",
        "image_id": 1,
        "annotations": [
          {"bbox": [x, y, w, h], "category_id": 0},
          ...
        ]
      },
      ...
    ]

Note: this project treats SKU-110K as a single-class density/counting
task (see src/training/train.py's ID2LABEL = {0: "product"}) --
every category_id in the output manifest is remapped to 0, regardless
of how many categories the source COCO file actually has. If you
later add per-SKU classification (docs/SDP.md Section 15, Future
Improvements), this remapping is the first place to change.

SUBSAMPLING (docs/SDP.md Section 6.3's scoping decision)
-------------------------------------------------------------
Images with ZERO annotations (an empty/near-empty shelf crop) are
kept -- they are a legitimate negative example, not something to
filter out (see src/data/dataset.py and src/data/augmentations.py,
both explicitly designed to handle this case).

Subsampling is done by RANDOM SAMPLING OF WHOLE IMAGES, never by
truncating a shuffled combined list across splits -- this preserves
Roboflow's own train/valid/test partitioning, which is what prevents
the same physical shelf/scene from leaking across splits (docs/SDP.md
Section 6.3's leakage warning).

HOW TO RUN (on Colab, after downloading/extracting the dataset)
-----------------------------------------------------------------------
    python scripts/build_manifests.py \\
        --dataset-root /content/data/raw/sku110k_export \\
        --output-dir data/splits \\
        --max-train 2500 --max-val 400 --max-test 400 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# This project's single-class taxonomy (see src/training/train.py's
# ID2LABEL/LABEL2ID) -- every source category_id maps to 0.
TARGET_CATEGORY_ID = 0


def load_coco_json(coco_json_path: Path) -> dict:
    """Load and do minimal structural validation on a COCO annotations file.

    Args:
        coco_json_path: Path to a COCO-format _annotations.coco.json file.

    Returns:
        The parsed COCO dict, guaranteed to have "images" and
        "annotations" keys (empty lists if genuinely absent).

    Raises:
        FileNotFoundError: If coco_json_path does not exist.
        ValueError: If the file doesn't parse as a dict, or is
            missing the "images" key entirely (a malformed/wrong-type
            export, not just an empty one).
    """
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO annotations file not found: {coco_json_path}")

    with coco_json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    if not isinstance(coco_data, dict):
        raise ValueError(
            f"Expected {coco_json_path} to contain a COCO-format JSON object, "
            f"got {type(coco_data).__name__} instead."
        )
    if "images" not in coco_data:
        raise ValueError(
            f"{coco_json_path} is missing the required 'images' key -- "
            f"this doesn't look like a valid COCO annotations file."
        )

    coco_data.setdefault("annotations", [])
    coco_data.setdefault("categories", [])
    return coco_data


def build_manifest_from_coco(coco_json_path: Path, images_dir: Path) -> list[dict]:
    """Convert one COCO annotations file into this project's manifest schema.

    Args:
        coco_json_path: Path to the split's _annotations.coco.json.
        images_dir: Directory containing the actual image files for
            this split (paths in the output manifest are written
            relative to this directory's parent, so they resolve
            correctly via src/data/dataset.py's PROJECT_ROOT-anchored
            path resolution).

    Returns:
        A list of manifest records, one per image, in the exact
        schema src/data/dataset.py's load_split() expects. Images
        with zero annotations are included with an empty
        "annotations" list (the "empty shelf" case), not dropped.

    Raises:
        FileNotFoundError: If coco_json_path or images_dir don't exist,
            or if an image file referenced in the COCO JSON is missing
            from images_dir.
        ValueError: If a COCO annotation is missing "bbox" or
            "category_id", or "bbox" doesn't have exactly 4 values --
            fail loudly here, at conversion time, rather than passing
            a malformed record through to load_split() to discover
            later (see src/data/dataset.py's own validation, which
            this script's checks deliberately mirror).
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    coco_data = load_coco_json(coco_json_path)

    # Group annotations by image_id up front -- O(n) instead of
    # re-scanning the full annotations list once per image (O(n*m)),
    # which matters once a real SKU-110K export has hundreds of
    # thousands of annotation rows.
    annotations_by_image_id: dict[int, list[dict]] = {}
    for ann in coco_data["annotations"]:
        if "bbox" not in ann:
            raise ValueError(f"Annotation missing 'bbox': {ann}")
        if "category_id" not in ann:
            raise ValueError(f"Annotation missing 'category_id': {ann}")
        if len(ann["bbox"]) != 4:
            raise ValueError(f"bbox must have exactly 4 values, got: {ann['bbox']!r}")

        annotations_by_image_id.setdefault(ann["image_id"], []).append(
            {
                "bbox": [float(v) for v in ann["bbox"]],
                # Remap every source category to this project's single
                # "product" class -- see this module's docstring.
                "category_id": TARGET_CATEGORY_ID,
            }
        )

    manifest = []
    for image_record in coco_data["images"]:
        file_name = image_record["file_name"]
        image_id = image_record["id"]
        image_path = images_dir / file_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced in {coco_json_path} not found on disk: {image_path}"
            )

        manifest.append(
            {
                "image_path": str(image_path),
                "image_id": image_id,
                "annotations": annotations_by_image_id.get(image_id, []),
            }
        )

    return manifest


def subsample_manifest(
    manifest: list[dict], max_images: int | None, seed: int
) -> list[dict]:
    """Randomly subsample a manifest to at most max_images records.

    Args:
        manifest: A full manifest, as produced by build_manifest_from_coco.
        max_images: Maximum number of images to keep. If None, or if
            it's >= len(manifest), the manifest is returned unchanged
            (order-shuffled, for good measure, but nothing dropped).
        seed: Random seed, for a reproducible subsample across re-runs
            of this script with the same arguments.

    Returns:
        A new list, containing at most max_images records.
    """
    rng = random.Random(seed)
    shuffled = manifest.copy()
    rng.shuffle(shuffled)

    if max_images is None or max_images >= len(shuffled):
        return shuffled
    return shuffled[:max_images]


def save_manifest(manifest: list[dict], output_path: Path) -> None:
    """Write a manifest to disk as JSON, creating parent directories if needed.

    Args:
        manifest: The manifest to write.
        output_path: Destination path, e.g. data/splits/train.json.

    Raises:
        ValueError: If manifest is empty -- an empty manifest would
            later fail src/data/dataset.py's load_split() anyway (by
            design, see that file's Common Mistakes section); failing
            here instead gives a clearer, earlier error message tied
            to the actual COCO export, not the downstream loader.
    """
    if len(manifest) == 0:
        raise ValueError(
            f"Refusing to write an empty manifest to {output_path} -- "
            f"check that the source COCO export and --max-* arguments "
            f"actually produced at least one record."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    print(f"Wrote {len(manifest)} records to {output_path}")


def build_all_splits(
    dataset_root: Path,
    output_dir: Path,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
    seed: int,
) -> None:
    """Convert all three splits (train/valid/test) and write manifests.

    Args:
        dataset_root: Root of the Roboflow COCO export (contains
            train/, valid/, test/ subfolders).
        output_dir: Where to write train.json, val.json, test.json
            (typically data/splits/).
        max_train: Max images to keep for the training split (see
            docs/SDP.md Section 6.3's scoping decision -- 2,000-3,000
            recommended).
        max_val: Max images to keep for the validation split.
        max_test: Max images to keep for the test split.
        seed: Random seed for reproducible subsampling.

    Returns:
        None. Writes train.json, val.json, and test.json to output_dir.
    """
    # Roboflow's default export folder name is "valid", not "val" --
    # this project's own naming convention is "val" (see
    # configs/training_config.yaml's val_manifest field) -- this
    # mapping is where that naming difference gets reconciled, in
    # exactly one place.
    split_folder_names = {"train": "train", "val": "valid", "test": "test"}
    max_images_by_split = {"train": max_train, "val": max_val, "test": max_test}

    for split_name, folder_name in split_folder_names.items():
        split_dir = dataset_root / folder_name
        coco_json_path = split_dir / "_annotations.coco.json"

        manifest = build_manifest_from_coco(coco_json_path, images_dir=split_dir)
        manifest = subsample_manifest(
            manifest, max_images_by_split[split_name], seed=seed
        )

        save_manifest(manifest, output_dir / f"{split_name}.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Convert a Roboflow COCO export of SKU-110K into "
        "this project's data/splits/*.json manifest files."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the extracted Roboflow export (contains train/, valid/, test/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Where to write train.json/val.json/test.json (default: data/splits).",
    )
    parser.add_argument(
        "--max-train", type=int, default=2500, help="Max training images to keep."
    )
    parser.add_argument(
        "--max-val", type=int, default=400, help="Max validation images to keep."
    )
    parser.add_argument(
        "--max-test", type=int, default=400, help="Max test images to keep."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_all_splits(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )

    # command to run the script:

# PS D:\smart-retail-shelf-auditing> python scripts\build_manifests.py --dataset-root "data\raw\shelf_dataset_source" --output-dir "data\splits" --max-train 4000 --max-val 500 --max-test 500
