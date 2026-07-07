"""
src.inference.inference

RESPONSIBILITY
--------------
Runs object detection on a single image using an already-loaded
RT-DETR model, and draws the resulting bounding boxes onto the image
for visualization. This is the ONE place raw model-in, boxes-out
logic lives — backend/routes/detect.py and backend/routes/report.py
both call into this module rather than duplicating the forward-pass
+ post-processing logic each.

ARCHITECTURE NOTES
-------------------
    • preprocess_image() and run_inference() reuse
      src/data/preprocessing.py's get_image_processor() — the same
      AutoImageProcessor instance used during training — so
      training/inference preprocessing never silently diverges.
    • This module has ZERO FastAPI dependency, by design (see this
      package's __init__.py).
    • draw_boxes_on_image() uses OpenCV, per docs/SDP.md Section 13's
      tech stack table: "OpenCV ... inference-time only, NOT
      annotation" — this is that one legitimate use.
    • run_inference() moves inputs to whatever device the model is
      already on (CPU or GPU) — this isn't explicitly called out in
      the original TODO, but is necessary for correctness: if the
      model was loaded onto a GPU (common after Colab training) and
      inputs stay on CPU, the forward pass raises a device-mismatch
      error. Device placement is inferred from the model itself
      rather than hardcoded, so this works identically in either
      environment with no config needed.

REFERENCES
----------
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr#transformers.RTDetrImageProcessor.post_process_object_detection
"""

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


def preprocess_image(image: Any, image_processor: Any) -> dict:
    """Preprocess a single raw image for inference (no annotations).

    Args:
        image: A PIL.Image in RGB mode.
        image_processor: An AutoImageProcessor instance (see
            src/data/preprocessing.py's get_image_processor()).

    Returns:
        A dict of model-ready tensors (at minimum "pixel_values"),
        suitable for passing directly to the model's forward call.
        Unlike src/data/preprocessing.py's preprocess_batch(), this
        takes no `annotations` argument — there are no ground-truth
        boxes at inference time.
    """
    return image_processor(images=image, return_tensors="pt")


def run_inference(
    image: Any,
    model: Any,
    image_processor: Any,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Run object detection on one image and return filtered detections.

    Args:
        image: A PIL.Image in RGB mode, at its original resolution.
        model: The loaded, .eval()-mode RT-DETR model (from
            backend/repositories/model_repository.py, or loaded
            directly for standalone/test usage).
        image_processor: The matching AutoImageProcessor.
        confidence_threshold: Minimum confidence score for a
            detection to be included in the returned list.

    Returns:
        A list of dicts, one per detected object, each with keys
        "x_min", "y_min", "x_max", "y_max" (original-image pixel
        coordinates), "confidence" (float), and "label" (str) —
        matching backend/schemas.py's BoundingBox fields exactly.
    """
    original_width, original_height = image.size  # PIL: (width, height)
    device = next(model.parameters()).device

    inputs = preprocess_image(image, image_processor)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes maps boxes back to ORIGINAL image coordinates, not
    # resized-image coordinates -- skipping this is the most common
    # bug here (see this module's earlier docstring notes).
    target_sizes = torch.tensor([[original_height, original_width]], device=device)

    # threshold is applied by the processor itself -- no separate
    # manual filtering step needed afterward.
    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=confidence_threshold,
    )[
        0
    ]  # batch of 1 -> take the single image's result

    id2label = getattr(model.config, "id2label", {}) or {}

    def _label_name(label_id: int) -> str:
        # id2label keys may be int or str depending on how the config
        # was loaded/serialized -- check both rather than assuming.
        if label_id in id2label:
            return id2label[label_id]
        if str(label_id) in id2label:
            return id2label[str(label_id)]
        return str(label_id)

    detections = []
    for box, score, label_id in zip(
        results["boxes"], results["scores"], results["labels"]
    ):
        x_min, y_min, x_max, y_max = box.tolist()
        detections.append(
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "confidence": float(score),
                "label": _label_name(int(label_id)),
            }
        )

    return detections


def draw_boxes_on_image(image: Any, detections: list[dict]) -> Any:
    """Draw bounding boxes and confidence scores onto a copy of the image.

    Args:
        image: The original PIL.Image the detections were computed on.
        detections: Output of run_inference().

    Returns:
        A NEW PIL.Image with each detection's box and confidence
        score drawn on top. The input image is not mutated.
    """
    # OpenCV works in BGR; convert once, draw, convert back -- this
    # keeps color tuples below meaning what they say (green box), not
    # silently swapped to blue/red because of channel-order confusion.
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    box_color = (60, 179, 30)  # BGR
    text_color = (255, 255, 255)

    for detection in detections:
        x_min = int(round(detection["x_min"]))
        y_min = int(round(detection["y_min"]))
        x_max = int(round(detection["x_max"]))
        y_max = int(round(detection["y_max"]))
        label = detection.get("label", "object")
        confidence = detection.get("confidence", 0.0)

        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), box_color, thickness=2)

        caption = f"{label} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(y_min - text_h - baseline - 4, 0)

        cv2.rectangle(
            image_bgr,
            (x_min, label_top),
            (x_min + text_w + 4, label_top + text_h + baseline + 4),
            box_color,
            thickness=-1,
        )
        cv2.putText(
            image_bgr,
            caption,
            (x_min + 2, label_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(annotated_rgb)
