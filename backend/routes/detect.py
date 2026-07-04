"""
backend.routes.detect

RESPONSIBILITY
--------------
Exposes POST /detect — accepts an uploaded shelf image and returns
raw detections (bounding boxes + confidence scores) from the
fine-tuned RT-DETR model. This route owns HTTP concerns ONLY
(request parsing, response shaping, status codes). It must NOT
contain inference logic itself.

PURPOSE
-------
This is the entry point that turns "a trained model sitting on disk"
into "a callable API." Everything downstream (report generation,
the frontend) depends on this endpoint's contract being stable.

ARCHITECTURE NOTES
-------------------
    • This route delegates actual model inference to
      src/inference/inference.py (imported here, called here, but
      IMPLEMENTED there) via the ModelRepository dependency.
    • Keep this file thin. If you find yourself writing more than a
      few lines of logic inside the route function itself, that
      logic almost certainly belongs in a service or the inference
      module instead.

ASCII FLOW DIAGRAM
-------------------
    Client
      |  POST /detect  (multipart/form-data image upload)
      v
    detect_products()                <- YOU implement this body
      |
      |  1. validate uploaded file (content-type, size)
      |  2. load image bytes -> PIL.Image
      |  3. call src.inference.inference.run_inference(image, model)
      |  4. wrap raw boxes into DetectionResponse
      v
    DetectionResponse  -->  returned to client as JSON

TODO
----
    - [ ] Implement detect_products() body:
          1. Validate `file.content_type` starts with "image/"
             (reject anything else with HTTPException(400, ...))
          2. Read file bytes and decode into a PIL.Image
          3. Call your inference function from src/inference/inference.py
             (that module is intentionally NOT implemented yet either
             — build it before wiring this endpoint end-to-end)
          4. Convert raw model output into a list[BoundingBox]
          5. Return a DetectionResponse

HINTS
-----
    - FastAPI's `UploadFile` gives you `.file`, `.filename`, and
      `.content_type` — use `await file.read()` to get raw bytes.
    - Consider a maximum upload size check before decoding the image
      to avoid loading huge files into memory unnecessarily.

COMMON MISTAKES
----------------
    - Doing image preprocessing (resize/normalize) directly in this
      route — that belongs in src/data/preprocessing.py or
      src/inference/inference.py, reused from training, not
      reimplemented here.
    - Forgetting to close/release the uploaded file handle — use
      `async with` or explicit `.close()` where appropriate.

BEST PRACTICES
---------------
    - Return meaningful HTTP status codes: 400 for bad input, 422
      for validation errors (FastAPI handles this for you via
      Pydantic automatically), 500 only for genuinely unexpected
      failures.

LEARNING NOTES
--------------
Compare this file's structure to routes/health.py once you're done —
same three-part pattern (route -> schema -> dependency), just with
real business logic instead of a trivial check.

REFERENCES
----------
    - https://fastapi.tiangolo.com/tutorial/request-files/
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.repositories.model_repository import ModelRepository, get_model_repository
from backend.schemas import DetectionResponse

router = APIRouter(tags=["detection"])


@router.post("/detect", response_model=DetectionResponse)
async def detect_products(
    file: UploadFile,
    repository: ModelRepository = Depends(get_model_repository),
) -> DetectionResponse:
    """Run object detection on an uploaded shelf image.

    Args:
        file: The uploaded image file (multipart/form-data).
        repository: Injected ModelRepository singleton providing
            access to the loaded RT-DETR model.

    Returns:
        DetectionResponse containing all detected bounding boxes,
        confidence scores, and image metadata.

    Raises:
        HTTPException: 400 if the uploaded file is not a valid
            image; 500 if inference fails unexpectedly.
    """
    # TODO: implement — see the ASCII flow diagram and TODO section
    # in this file's module docstring for the exact steps.
    raise NotImplementedError("detect_products() is not implemented yet")
