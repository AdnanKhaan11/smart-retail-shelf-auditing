"""
backend.routes.report

RESPONSIBILITY
--------------
Exposes POST /report — accepts an uploaded shelf image (and an
optional baseline "full shelf" product count) and returns a
structured inventory report: detected count, stock-level percentage,
and a low-stock flag. This route owns HTTP concerns ONLY; all
counting/thresholding logic belongs in
backend/services/inventory.py.

PURPOSE
-------
This is the endpoint that turns raw detections into a *business*
signal a store manager could actually act on ("Aisle 4, Shelf B is
at 32% stock — restock needed"). This is the file that makes the
project a product rather than just a model demo — treat it that way.

ARCHITECTURE NOTES
-------------------
    • This route calls detect_products()'s underlying inference path
      (or duplicates the minimal call to the model — decide and
      document which, to avoid two divergent code paths) AND the
      InventoryService (backend/services/inventory.py) to turn boxes
      into a report.
    • Do not reimplement counting/threshold logic here — call
      InventoryService methods instead. If you find yourself writing
      `if count < threshold:` directly in this file, stop and move
      that logic into the service.

ASCII FLOW DIAGRAM
-------------------
    Client
      |  POST /report  (image + optional baseline_count)
      v
    generate_inventory_report()       <- YOU implement this body
      |
      |  1. run detection (reuse inference path from detect.py)
      |  2. pass detections + baseline_count to InventoryService
      |  3. wrap the service's result into InventoryReportResponse
      v
    InventoryReportResponse  -->  returned to client as JSON

TODO
----
    - [ ] Implement generate_inventory_report() body:
          1. Reuse the same inference call as detect.py (consider
             extracting a small shared helper if you find yourself
             copy-pasting the same 4 lines in both routes)
          2. Instantiate/inject InventoryService and call its
             generate_report() method with the detections and a
             baseline_count (from the request, or a sane default —
             your call, document the decision)
          3. Return an InventoryReportResponse

HINTS
-----
    - Consider accepting `baseline_count` as an optional query
      parameter with a documented default, rather than requiring the
      client to always supply it.

COMMON MISTAKES
----------------
    - Putting the low-stock threshold value directly in this route
      as a magic number — it belongs in configs/inference_config.yaml
      and should be read by InventoryService's constructor, not
      hardcoded here.

BEST PRACTICES
---------------
    - Keep this route a thin orchestrator: call inference, call the
      service, return the schema. If it grows past ~20-30 lines,
      logic has leaked in that belongs elsewhere.

LEARNING NOTES
--------------
This route is a good place to notice the difference between a
"route" (HTTP shape) and a "service" (business rules) — the same
distinction shows up in almost every production backend you'll ever
work on.

REFERENCES
----------
    - https://fastapi.tiangolo.com/tutorial/query-params/
"""

from fastapi import APIRouter, Depends, UploadFile

from backend.repositories.model_repository import ModelRepository, get_model_repository
from backend.schemas import InventoryReportResponse
from backend.services.inventory import InventoryService, get_inventory_service

router = APIRouter(tags=["inventory"])


@router.post("/report", response_model=InventoryReportResponse)
async def generate_inventory_report(
    file: UploadFile,
    baseline_count: int | None = None,
    repository: ModelRepository = Depends(get_model_repository),
    inventory_service: InventoryService = Depends(get_inventory_service),
) -> InventoryReportResponse:
    """Generate a structured inventory report for an uploaded shelf image.

    Args:
        file: The uploaded shelf image (multipart/form-data).
        baseline_count: Optional expected "full shelf" product count
            used to compute stock-level percentage. If omitted, the
            service should fall back to a documented default (see
            InventoryService).
        repository: Injected ModelRepository singleton.
        inventory_service: Injected InventoryService singleton.

    Returns:
        InventoryReportResponse containing detected_count,
        stock_level_percent, is_low_stock, and supporting detail.

    Raises:
        HTTPException: 400 if the uploaded file is not a valid
            image; 500 if inference or report generation fails.
    """
    # TODO: implement — see the ASCII flow diagram and TODO section
    # in this file's module docstring for the exact steps.
    raise NotImplementedError("generate_inventory_report() is not implemented yet")
