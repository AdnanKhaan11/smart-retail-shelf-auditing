"""
backend.services.inventory

RESPONSIBILITY
--------------
Contains ALL business logic for turning raw object-detection output
into an actionable inventory signal: counting products, computing a
stock-level percentage against a baseline "full shelf" count, and
flagging shelves that need restocking.

PURPOSE
-------
This is arguably the single most important file in the whole
backend for your resume story — it's the piece that turns "a model
that draws boxes" into "a system that tells a manager what to do."
Recruiters and interviewers will ask about this file specifically;
make sure you can explain every decision made here.

ARCHITECTURE NOTES
-------------------
    • This class must be FRAMEWORK-AGNOSTIC. No FastAPI imports here
      at all. That is what lets you unit-test it directly (see
      tests/test_inventory.py) without spinning up an HTTP server,
      and it's also what would let you reuse this exact logic from a
      CLI tool or a batch job later without any changes.
    • The low-stock threshold is a CONFIGURATION value, not a magic
      number — it must be injected via the constructor (read from
      configs/inference_config.yaml at startup), never hardcoded
      inside a method.

ASCII FLOW DIAGRAM
-------------------
    list[BoundingBox]  (raw detections from the model)
            |
            v
    count_detections()          -> int
            |
            v
    compute_stock_level()       -> float (0.0 - 1.0, or 0-100%)
            |
            v
    is_low_stock()               -> bool
            |
            v
    generate_report()            -> dict / InventoryReportResponse-ready data

TODO
----
    - [ ] Implement __init__ to accept and store `low_stock_threshold`
          (e.g. 0.6 meaning "below 60% of baseline counts as low stock")
    - [ ] Implement count_detections() — this one is genuinely almost
          trivial (len() of the input list) but keep it as its own
          method rather than inlining len(detections) everywhere,
          because "what counts as a valid detection" may later grow
          more complex (e.g. filtering by a minimum confidence
          threshold) and you want exactly one place to change that.
    - [ ] Implement compute_stock_level() — decide: percentage
          (0-100) or ratio (0.0-1.0)? Be consistent with your
          schemas.py InventoryReportResponse field naming/type.
    - [ ] Implement is_low_stock() — a simple threshold comparison,
          but make sure it uses self.low_stock_threshold, not a
          hardcoded number.
    - [ ] Implement generate_report() — orchestrates the three
          methods above into a single dict/object ready to be
          wrapped in InventoryReportResponse by the route.

HINTS
-----
    - Think about what "baseline_count" should default to if the
      caller doesn't supply one. Zero? The max ever observed for
      this shelf? Document whatever you choose and why — this is
      exactly the kind of product decision an interviewer will ask
      you to justify.
    - Consider whether you want to filter out low-confidence
      detections (e.g. confidence < 0.5) before counting — this is a
      precision/recall tradeoff decision (see Section 11 of
      docs/SDP.md) that belongs here, not in the route.

COMMON MISTAKES
----------------
    - Putting HTTP status codes or exceptions meant for API responses
      inside this class — raise plain Python exceptions here if
      something is genuinely invalid (e.g. negative baseline_count),
      and let the route translate that into an HTTPException.
    - Hardcoding the low-stock threshold instead of injecting it via
      the constructor — this makes the threshold impossible to tune
      without a code change, which defeats the purpose of having it
      configurable at all.

BEST PRACTICES
---------------
    - Keep each method doing exactly one thing (Single Responsibility
      Principle) — resist the urge to merge count + threshold logic
      into one big method "for convenience."
    - Write tests/test_inventory.py against this class directly,
      with plain Python lists of fake detections — no FastAPI, no
      model, no image files needed to test business logic.

LEARNING NOTES
--------------
This file is a clean example of Clean Architecture's "use case" /
"interactor" layer — logic that represents what the SYSTEM DOES,
independent of how it's triggered (HTTP here, but could be a CLI,
a scheduled job, anything) or where its inputs came from (a real
model here, but could be a mocked list of boxes in a test).

REFERENCES
----------
    - Clean Architecture, Robert C. Martin — the "use case" layer
      concept this file implements.
"""

from dataclasses import dataclass


@dataclass
class InventoryService:
    """Transforms raw detections into an actionable inventory report.

    Attributes:
        low_stock_threshold: Fraction (0.0-1.0) of baseline_count
            below which a shelf is considered low-stock. Should be
            loaded from configs/inference_config.yaml at application
            startup, not hardcoded at call sites.
    """

    low_stock_threshold: float = 0.6

    def count_detections(self, detections: list[dict]) -> int:
        """Count the number of valid product detections.

        Args:
            detections: Raw detection dicts (or BoundingBox-like
                objects) produced by the inference pipeline.

        Returns:
            The number of detections counted as "present" for
            inventory purposes.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("count_detections() is not implemented yet")

    def compute_stock_level(self, detected_count: int, baseline_count: int) -> float:
        """Compute the current stock level relative to a baseline.

        Args:
            detected_count: Number of products currently detected.
            baseline_count: Expected number of products on a fully
                stocked shelf.

        Returns:
            Stock level as a value between 0.0 and 1.0 (or 0-100 if
            you decide to express it as a percentage — be consistent
            with schemas.py).

        Raises:
            NotImplementedError: Always, until implemented.
            ValueError: Should be raised (once implemented) if
                baseline_count <= 0, since that would be undefined.
        """
        raise NotImplementedError("compute_stock_level() is not implemented yet")

    def is_low_stock(self, stock_level: float) -> bool:
        """Determine whether a stock level counts as "needs restocking."

        Args:
            stock_level: Value returned by compute_stock_level().

        Returns:
            True if stock_level is below self.low_stock_threshold.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("is_low_stock() is not implemented yet")

    def generate_report(self, detections: list[dict], baseline_count: int) -> dict:
        """Orchestrate counting, stock-level, and low-stock flagging.

        Args:
            detections: Raw detection dicts from the inference pipeline.
            baseline_count: Expected full-shelf product count.

        Returns:
            A dict shaped to match InventoryReportResponse's fields
            once schemas.InventoryReportResponse is implemented.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("generate_report() is not implemented yet")


def get_inventory_service() -> InventoryService:
    """FastAPI dependency provider for InventoryService.

    Returns:
        An InventoryService instance. TODO: once
        configs/inference_config.yaml is implemented, load
        low_stock_threshold from it here instead of relying on the
        dataclass default.
    """
    # TODO: read low_stock_threshold from configs/inference_config.yaml
    return InventoryService()
