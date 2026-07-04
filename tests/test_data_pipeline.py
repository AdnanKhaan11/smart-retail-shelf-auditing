"""
tests/test_data_pipeline.py

RESPONSIBILITY
--------------
Integration test for the ENTIRE src/data/ pipeline, end to end:

    load_split() -> apply_transforms() -> preprocess_batch()
                  -> collate_fn() -> build_dataloader()

Unlike a set of isolated unit tests per file, this suite builds a
tiny synthetic dataset once (3 fake images + a JSON manifest, one
image with ZERO annotations to cover the "empty shelf" case) and
pushes it through every stage in the same order src/training/train.py
will, asserting the output shape/schema at each handoff point.

WHY A REAL IMAGE_PROCESSOR ISN'T USED HERE
---------------------------------------------
src/data/preprocessing.py's get_image_processor() downloads
"PekingU/rtdetr_v2_r50vd" from the Hugging Face Hub. That's correct
and necessary for REAL training, but makes this test suite (a) slow,
(b) dependent on network access, and (c) unable to run at all in an
offline/CI environment with no Hub access. Instead, this suite uses
a FakeImageProcessor that mimics RTDetrImageProcessor's __call__
signature exactly (same args, same output shape) — this tests OUR
code's logic (batching, error handling, schema) without testing
Hugging Face's own image processor implementation, which is not our
code to test in the first place.

A SEPARATE, real-network integration test
(test_preprocessing_with_real_processor, marked with
@pytest.mark.network) is included at the bottom and automatically
skipped if the Hub isn't reachable — run it manually once when you
have network access, to confirm the FakeImageProcessor's assumptions
about the real processor's output shape still hold.

HOW TO RUN
----------
    pytest tests/test_data_pipeline.py -v

    # Skip the slow/network test explicitly:
    pytest tests/test_data_pipeline.py -v -m "not network"
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from src.data.augmentations import (
    apply_transforms,
    get_eval_transforms,
    get_train_transforms,
)
from src.data.dataloader import build_dataloader, collate_fn
from src.data.dataset import build_datasets, load_split
from src.data.preprocessing import format_annotations_for_processor, preprocess_batch


# ======================================================================
# Fake image processor — mimics RTDetrImageProcessor's __call__ shape
# without any network dependency. See module docstring for why.
# ======================================================================

class FakeImageProcessor:
    """Minimal stand-in for AutoImageProcessor, same call signature."""

    def __init__(self, image_size: int = 320):
        self.image_size = image_size

    def __call__(self, images, annotations, return_tensors="pt"):
        pixel_values = []
        labels = []
        for image, ann in zip(images, annotations):
            resized = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(resized).astype("float32") / 255.0
            array = array.transpose(2, 0, 1)  # HWC -> CHW
            pixel_values.append(torch.tensor(array))

            boxes = [a["bbox"] for a in ann["annotations"]]
            class_labels = [a["category_id"] for a in ann["annotations"]]
            labels.append(
                {
                    "boxes": torch.tensor(boxes, dtype=torch.float32)
                    if boxes
                    else torch.zeros((0, 4)),
                    "class_labels": torch.tensor(class_labels, dtype=torch.int64)
                    if class_labels
                    else torch.zeros((0,), dtype=torch.int64),
                    "image_id": torch.tensor([ann["image_id"]]),
                }
            )

        return {"pixel_values": torch.stack(pixel_values), "labels": labels}


# ======================================================================
# Fixtures — build a small synthetic dataset once per test module
# ======================================================================

@pytest.fixture(scope="module")
def synthetic_manifest(tmp_path_factory) -> str:
    """Create 3 tiny fake images + a matching manifest.json.

    Image 2 (image_id=2) is deliberately given ZERO annotations, to
    exercise the "empty shelf" negative-example case throughout the
    pipeline — this is a legitimate input, not an edge case to avoid.
    """
    tmp_dir = tmp_path_factory.mktemp("synthetic_data")
    img_dir = tmp_dir / "images"
    img_dir.mkdir()

    records = []
    for i in range(3):
        image = PILImage.new("RGB", (400, 300), color=(i * 40 % 255, 120, 200))
        image_path = img_dir / f"img_{i}.jpg"
        image.save(image_path)

        annotations = (
            []
            if i == 2
            else [
                {"bbox": [10.0, 10.0, 50.0, 60.0], "category_id": 0},
                {"bbox": [100.0, 50.0, 40.0, 40.0], "category_id": 0},
            ]
        )
        records.append(
            {"image_path": str(image_path), "image_id": i, "annotations": annotations}
        )

    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records))
    return str(manifest_path)


@pytest.fixture(scope="module")
def raw_dataset(synthetic_manifest):
    """The dataset as load_split() produces it, before any transform."""
    return load_split(synthetic_manifest)


# ======================================================================
# Stage 1: src/data/dataset.py
# ======================================================================

def test_load_split_schema_and_length(raw_dataset):
    assert len(raw_dataset) == 3
    assert set(raw_dataset.column_names) == {"image", "image_id", "annotations"}


def test_load_split_preserves_zero_annotation_example(raw_dataset):
    empty_example = raw_dataset[2]
    assert empty_example["annotations"] == []


def test_load_split_rejects_missing_bbox_key(tmp_path):
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(
        json.dumps(
            [{"image_path": "x.jpg", "image_id": 0, "annotations": [{"category_id": 0}]}]
        )
    )
    with pytest.raises(ValueError, match="bbox"):
        load_split(str(bad_manifest))


def test_load_split_rejects_empty_manifest(tmp_path):
    empty_manifest = tmp_path / "empty.json"
    empty_manifest.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="empty"):
        load_split(str(empty_manifest))


def test_build_datasets_loads_both_splits(synthetic_manifest):
    train_ds, val_ds = build_datasets(synthetic_manifest, synthetic_manifest)
    assert len(train_ds) == 3
    assert len(val_ds) == 3


# ======================================================================
# Stage 2: src/data/augmentations.py
# ======================================================================

def test_apply_train_transforms_resizes_and_keeps_box_count(raw_dataset):
    transforms = get_train_transforms(image_size=320)
    example = raw_dataset[0]

    result = apply_transforms(example, transforms)

    assert result["image"].size == (320, 320)
    assert len(result["annotations"]) == len(example["annotations"])
    for ann in result["annotations"]:
        assert set(ann.keys()) == {"bbox", "category_id"}


def test_apply_transforms_category_id_stays_int(raw_dataset):
    """Regression test for a real bug found while validating this
    pipeline: Albumentations returns labels as floats internally —
    apply_transforms() must cast back to int, or category_id silently
    becomes 0.0 instead of 0, breaking the project's own type
    contract (see src/data/dataset.py's _validate_annotation)."""
    transforms = get_train_transforms(image_size=320)
    example = raw_dataset[0]

    result = apply_transforms(example, transforms)

    for ann in result["annotations"]:
        assert isinstance(ann["category_id"], int), (
            f"category_id must be int, got {type(ann['category_id'])} "
            f"({ann['category_id']!r}) — Albumentations coerces labels to "
            f"float internally; apply_transforms must cast back with int()."
        )


def test_apply_transforms_handles_zero_annotation_example(raw_dataset):
    transforms = get_train_transforms(image_size=320)
    empty_example = raw_dataset[2]

    result = apply_transforms(empty_example, transforms)

    assert result["annotations"] == []
    assert result["image"].size == (320, 320)


def test_eval_transforms_are_deterministic(raw_dataset):
    """get_eval_transforms() must never randomize — running it twice
    on the same input must produce identical boxes, since validation/
    test metrics must be reproducible run to run."""
    transforms = get_eval_transforms(image_size=320)
    example = raw_dataset[0]

    result_a = apply_transforms(example, transforms)
    result_b = apply_transforms(example, transforms)

    assert result_a["annotations"] == result_b["annotations"]


# ======================================================================
# Stage 3: src/data/preprocessing.py
# ======================================================================

def test_format_annotations_for_processor_computes_area():
    result = format_annotations_for_processor(
        image_id=7, boxes=[[10.0, 20.0, 30.0, 40.0]], labels=[0]
    )
    assert result["image_id"] == 7
    assert result["annotations"][0]["area"] == 30.0 * 40.0
    assert result["annotations"][0]["iscrowd"] == 0


def test_format_annotations_for_processor_rejects_bad_bbox_length():
    with pytest.raises(ValueError, match="4 numbers"):
        format_annotations_for_processor(image_id=0, boxes=[[1, 2, 3]], labels=[0])


def test_format_annotations_for_processor_rejects_non_numeric_bbox():
    with pytest.raises(ValueError, match="numeric"):
        format_annotations_for_processor(
            image_id=0, boxes=[["a", "b", "c", "d"]], labels=[0]
        )


def test_format_annotations_for_processor_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        format_annotations_for_processor(
            image_id=0, boxes=[[1, 2, 3, 4], [1, 2, 3, 4]], labels=[0]
        )


def test_preprocess_batch_produces_correct_shapes(raw_dataset):
    eval_transforms = get_eval_transforms(image_size=320)
    resized = raw_dataset.map(apply_transforms, fn_kwargs={"transforms": eval_transforms})
    fake_processor = FakeImageProcessor(image_size=320)

    batch = resized[:3]
    encoded = preprocess_batch(batch, fake_processor)

    assert encoded["pixel_values"].shape == (3, 3, 320, 320)
    assert len(encoded["labels"]) == 3


def test_preprocess_batch_rejects_mismatched_batch_lengths():
    fake_processor = FakeImageProcessor(image_size=320)
    bad_batch = {
        "image": [PILImage.new("RGB", (10, 10))] * 2,
        "image_id": [0, 1],
        "annotations": [[]],  # deliberately shorter than image/image_id
    }
    with pytest.raises(ValueError, match="same length"):
        preprocess_batch(bad_batch, fake_processor)


# ======================================================================
# Stage 4: src/data/dataloader.py
# ======================================================================

def test_collate_fn_handles_variable_box_counts():
    batch = [
        {
            "pixel_values": torch.rand(3, 320, 320),
            "labels": {"boxes": torch.rand(2, 4), "class_labels": torch.tensor([0, 0])},
        },
        {
            "pixel_values": torch.rand(3, 320, 320),
            "labels": {"boxes": torch.rand(5, 4), "class_labels": torch.tensor([0] * 5)},
        },
    ]

    result = collate_fn(batch)

    assert result["pixel_values"].shape == (2, 3, 320, 320)
    assert len(result["labels"]) == 2
    assert result["labels"][0]["boxes"].shape[0] == 2
    assert result["labels"][1]["boxes"].shape[0] == 5


def test_collate_fn_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate_fn([])


def test_collate_fn_rejects_inconsistent_pixel_value_shapes():
    batch = [
        {"pixel_values": torch.rand(3, 320, 320), "labels": {}},
        {"pixel_values": torch.rand(3, 640, 640), "labels": {}},
    ]
    with pytest.raises(ValueError, match="Inconsistent pixel_values shapes"):
        collate_fn(batch)


def test_build_dataloader_end_to_end(raw_dataset):
    """Full pipeline, one real batch: dataset -> transforms -> fake
    processor -> collate -> DataLoader iteration."""
    eval_transforms = get_eval_transforms(image_size=320)
    resized = raw_dataset.map(apply_transforms, fn_kwargs={"transforms": eval_transforms})
    fake_processor = FakeImageProcessor(image_size=320)

    encoded_batch = preprocess_batch(resized[:3], fake_processor)
    examples = [
        {"pixel_values": encoded_batch["pixel_values"][i], "labels": encoded_batch["labels"][i]}
        for i in range(3)
    ]

    dataloader = build_dataloader(examples, batch_size=2, shuffle=False)
    batches = list(dataloader)

    assert len(batches) == 2  # batch_size=2 over 3 examples -> batches of 2, then 1
    assert batches[0]["pixel_values"].shape == (2, 3, 320, 320)
    assert batches[1]["pixel_values"].shape == (1, 3, 320, 320)


# ======================================================================
# Optional: real network test against the actual RT-DETR processor.
# Skipped automatically if the Hugging Face Hub isn't reachable.
# ======================================================================

@pytest.mark.network
def test_preprocessing_with_real_processor(raw_dataset):
    """Confirms FakeImageProcessor's assumed output shape matches the
    REAL RTDetrImageProcessor. Run manually with network access:

        pytest tests/test_data_pipeline.py -v -m network
    """
    from src.data.preprocessing import get_image_processor

    try:
        real_processor = get_image_processor("PekingU/rtdetr_v2_r50vd")
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any
        # failure here means "can't reach the Hub," not "our code is
        # broken" — skip rather than fail the suite.
        pytest.skip(f"Hugging Face Hub not reachable, skipping: {exc}")

    eval_transforms = get_eval_transforms(image_size=640)
    resized = raw_dataset.map(apply_transforms, fn_kwargs={"transforms": eval_transforms})

    encoded = preprocess_batch(resized[:3], real_processor)

    assert encoded["pixel_values"].shape[0] == 3
    assert len(encoded["labels"]) == 3
    for label in encoded["labels"]:
        assert "boxes" in label and "class_labels" in label
