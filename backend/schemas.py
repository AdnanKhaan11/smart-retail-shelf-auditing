"""
backend.schemas
================

RESPONSIBILITY
--------------
Defines every Pydantic request/response model used by the API. This
is the single source of truth for "what shape does data have when it
crosses the API boundary" — routes, services, and tests should all
import types from here rather than redefining dicts ad hoc.

PURPOSE
-------
Pydantic schemas give FastAPI automatic request validation, response
serialization, and OpenAPI (Swagger) documentation for free. A field
you forget to declare here is a field FastAPI will silently drop from
responses or reject on requests — so this file matters more than its
small size suggests.

ARCHITECTURE NOTES
-------------------
    • Schemas here describe API I/O shape ONLY. They must never
      contain business logic (no counting, no thresholding, no
      report generation) — that belongs in backend/services/.
    • Keep schemas flat and explicit. Resist reusing a single "God
      schema" for multiple endpoints — each endpoint gets its own
      request/response model, even if they're similar, because
      endpoints evolve independently over time.

ASCII FLOW DIAGRAM
-------------------
    Client Upload (raw bytes)
            |
            v
    backend/routes/detect.py   <-- validates OUTPUT against DetectionResponse
            |
            v
    backend/services/inventory.py
            |
            v
    backend/routes/report.py   <-- validates OUTPUT against InventoryReportResponse

TODO
----
    - [ ] Implement DetectionResponse (pattern-match BoundingBox below)
    - [ ] Implement InventoryReportResponse
    - [ ] Implement ErrorResponse (used by FastAPI exception handlers)
    - [ ] Consider adding a `model_config = ConfigDict(frozen=True)`
          to response models once your fields stabilize, to prevent
          accidental mutation after construction

HINTS
-----
    - Use `Field(..., description="...")` on every field — it shows
      up in your auto-generated Swagger docs at /docs, which is free
      documentation you should not skip.
    - Confidence scores are floats in [0.0, 1.0] — consider using
      `Field(ge=0.0, le=1.0)` to get validation for free.

COMMON MISTAKES
----------------
    - Making every field Optional "just in case" — this silently
      hides bugs where a field should always be present. Only mark a
      field Optional if there is a real, documented reason a client
      might omit it.
    - Returning raw dicts from routes instead of schema instances —
      you lose validation and OpenAPI docs if you do this.

BEST PRACTICES
---------------
    - One schema per distinct API shape, not one shared "generic"
      schema reused everywhere.
    - Name response schemas with a `Response` suffix and request
      schemas with a `Request` suffix for immediate readability.

LEARNING NOTES
--------------
BoundingBox below is fully implemented as your reference pattern.
Every other schema in this file should follow the exact same shape:
inherit BaseModel, declare typed fields with Field(...) metadata, add
a short class docstring.

REFERENCES
----------
    - https://docs.pydantic.dev/latest/concepts/models/
    - https://fastapi.tiangolo.com/tutorial/response-model/
"""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """A single detected product's bounding box and confidence.

    This class is fully implemented as your reference pattern for
    every other schema in this file — copy this structure, don't
    reinvent it.

    Attributes:
        x_min: Left edge of the box, in pixels, relative to the
            original (un-resized) uploaded image.
        y_min: Top edge of the box, in pixels.
        x_max: Right edge of the box, in pixels.
        y_max: Bottom edge of the box, in pixels.
        confidence: Model confidence score for this detection,
            in the range [0.0, 1.0].
        label: Class label for this detection. For this project
            there is effectively one class ("product"), but the
            field is kept generic for future multi-class work
            (see Future Improvements in docs/SDP.md).
    """

    x_min: float = Field(..., description="Left edge of box in pixels")
    y_min: float = Field(..., description="Top edge of box in pixels")
    x_max: float = Field(..., description="Right edge of box in pixels")
    y_max: float = Field(..., description="Bottom edge of box in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    label: str = Field(default="product", description="Detected class label")


class DetectionResponse(BaseModel):
    """Response schema for POST /detect.

    TODO: Implement this class following the BoundingBox pattern
    above. It should describe everything the /detect endpoint
    returns to the client after running inference on one image.

    Suggested fields (confirm against your actual inference.py
    output before finalizing):
        - boxes: list[BoundingBox]
        - image_width: int
        - image_height: int
        - inference_time_ms: float
        - annotated_image_base64: str | None (if you choose to
          return the drawn-on image directly rather than just boxes)
    """

    # TODO: implement fields
    ...


class InventoryReportResponse(BaseModel):
    """Response schema for POST /report.

    TODO: Implement this class. It should describe the structured
    inventory report produced by backend/services/inventory.py —
    see that file's docstring for the exact business fields you'll
    need to surface here (detected_count, baseline_count,
    stock_level_percent, is_low_stock, etc.).
    """

    # TODO: implement fields
    ...


class HealthResponse(BaseModel):
    """Response schema for GET /health.

    Fully implemented — this is intentionally trivial and used
    directly by backend/routes/health.py, which is also fully
    implemented as your second reference example.

    Attributes:
        status: Literal liveness indicator, always "ok" if the
            process is able to respond at all.
        model_loaded: Whether the detection model has finished
            loading in backend/repositories/model_repository.py.
    """

    status: str = Field(default="ok", description="Liveness indicator")
    model_loaded: bool = Field(..., description="Whether the model is ready for inference")


class ErrorResponse(BaseModel):
    """Standard error response shape for FastAPI exception handlers.

    TODO: Implement this class. Keep it simple and consistent — every
    error your API returns (400, 422, 500, etc.) should be shaped
    the same way so frontend/app.py can handle errors generically
    instead of special-casing each endpoint.

    Suggested fields:
        - detail: str
        - error_code: str | None
    """

    # TODO: implement fields
    ...
