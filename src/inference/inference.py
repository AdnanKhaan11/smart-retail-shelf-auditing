"""
src.inference.inference

RESPONSIBILITY
--------------
Runs object detection on a single image using an already-loaded
RT-DETR model, and (optionally) draws the resulting bounding boxes
onto the image for visualization. This is the ONE place raw
model-in, boxes-out logic lives — backend/routes/detect.py and
backend/routes/report.py both call into this module rather than
duplicating the forward-pass + post-processing logic each.

PURPOSE
-------
This file is what backend/routes/detect.py's TODO explicitly points
to (see that file's module docstring, step 3: "call
src.inference.inference.run_inference"). Implementing it correctly
here, once, means both API endpoints (and any future CLI/batch
tooling you build) share identical, consistent inference behavior.

ARCHITECTURE NOTES
-------------------
    • run_inference() should reuse the SAME image preprocessing path
      used during training (src/data/preprocessing.py's
      get_image_processor(), and ideally get_eval_transforms() from
      src/data/augmentations.py for the deterministic resize) —
      training/inference preprocessing mismatch is one of the most
      common silent sources of degraded real-world accuracy (see
      src/data/preprocessing.py's module docstring for more on this).
    • This module must remain importable and runnable with ZERO
      FastAPI dependency (see this package's __init__.py).
    • draw_boxes_on_image() is the one legitimate use of OpenCV in
      this project (per docs/SDP.md Section 13's tech stack table:
      "OpenCV ... inference-time only, NOT annotation").

ASCII FLOW DIAGRAM
-------------------
    PIL.Image (from an uploaded file, decoded by backend/routes/detect.py)
            |
            v
    preprocess_image(image, image_processor)   <- YOU implement this
            |   (reuses src/data/preprocessing.py's conversion logic —
            |    do NOT reimplement it here)
            v
    model(**inputs)   (forward pass — this part is trivial once
                        preprocessing is correct)
            |
            v
    image_processor.post_process_object_detection(...)
            |
            v
    run_inference() returns: list[{"x_min":.., "y_min":.., "x_max":..,
                                    "y_max":.., "confidence":.., "label":..}]
            |
            v
    (optional) draw_boxes_on_image(image, detections) -> annotated PIL.Image

TODO
----
    - [ ] Implement preprocess_image(image, image_processor):
          Reuse src.data.preprocessing's conversion approach, but
          note there are no ground-truth annotations at inference
          time — this is a simpler, annotation-free preprocessing
          path (image only, no boxes to convert), NOT a call to
          preprocess_batch() (which expects annotations).
    - [ ] Implement run_inference(image, model, image_processor,
          confidence_threshold=0.5):
          1. Call preprocess_image() to get model-ready tensors
          2. Run model(**inputs) (remember: model must already be in
             .eval() mode — that's ModelRepository's responsibility,
             not this function's, but double-check it if you see odd
             results)
          3. Call image_processor.post_process_object_detection()
             with the ORIGINAL (pre-resize) image size, so returned
             boxes are in original-image pixel coordinates, not
             resized-image coordinates — this is a very common bug
          4. Filter by confidence_threshold
          5. Return a list of dicts matching backend/schemas.py's
             BoundingBox field names exactly (x_min, y_min, x_max,
             y_max, confidence, label) so the route layer can
             construct BoundingBox instances directly without any
             reshaping
    - [ ] Implement draw_boxes_on_image(image, detections):
          Use OpenCV (cv2.rectangle, cv2.putText) or PIL's
          ImageDraw — either is fine, but be consistent — to draw
          each detection's box and confidence score onto a copy of
          the original image. Return a NEW image; do not mutate the
          input in place.

HINTS
-----
    - `image_processor.post_process_object_detection(outputs,
      target_sizes=[(original_height, original_width)],
      threshold=confidence_threshold)` is the standard call —
      target_sizes is what maps boxes back to original image
      coordinates; get this wrong and every box will be scaled
      incorrectly.
    - Test this function on one known image with a hand-verified
      expected detection before wiring it into the API end to end.

COMMON MISTAKES
----------------
    - Returning boxes in RESIZED-image coordinates instead of
      ORIGINAL-image coordinates — target_sizes in
      post_process_object_detection is what prevents this; skipping
      it is a very easy mistake to make and produces boxes that look
      "almost right but scaled wrong" when drawn.
    - Forgetting model.eval() / torch.no_grad() context around the
      forward pass — impacts both correctness (dropout/batchnorm
      behavior) and unnecessary memory usage from tracked gradients
      you'll never use at inference time.

BEST PRACTICES
---------------
    - Keep this module's return shape (list of plain dicts with
      exact field names matching BoundingBox) stable — it is the
      informal "contract" between this module and the backend route
      layer.

LEARNING NOTES
--------------
Once implemented, compare this file's preprocessing path against
src/data/preprocessing.py's — they should be doing conceptually the
same image transform (resize + normalize), just with training-only
concerns (augmentation, annotation formatting) stripped out. If they
diverge in HOW they resize/normalize, that mismatch will show up as
a real, hard-to-diagnose accuracy gap between your reported training
metrics and real-world demo performance.

REFERENCES
----------
    - https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr#transformers.RTDetrImageProcessor.post_process_object_detection
"""

from typing import Any


def preprocess_image(image: Any, image_processor: Any) -> dict:
    """Preprocess a single raw image for inference (no annotations).

    Args:
        image: A PIL.Image in RGB mode.
        image_processor: An AutoImageProcessor instance (see
            src/data/preprocessing.py's get_image_processor()).

    Returns:
        A dict of model-ready tensors (at minimum "pixel_values"),
        suitable for passing directly to the model's forward call.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("preprocess_image() is not implemented yet")


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

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("run_inference() is not implemented yet")


def draw_boxes_on_image(image: Any, detections: list[dict]) -> Any:
    """Draw bounding boxes and confidence scores onto a copy of the image.

    Args:
        image: The original PIL.Image the detections were computed on.
        detections: Output of run_inference().

    Returns:
        A NEW PIL.Image (or numpy array, if you choose to work in
        OpenCV's native format) with each detection's box and
        confidence score drawn on top. The input image must not be
        mutated.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("draw_boxes_on_image() is not implemented yet")
