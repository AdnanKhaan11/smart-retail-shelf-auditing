"""
backend.services
=================

RESPONSIBILITY
--------------
Marks `backend/services/` as a package. Services hold BUSINESS
LOGIC — rules about the domain (counting, thresholds, report shape)
that have nothing to do with HTTP or the model itself.

WHAT BELONGS HERE
------------------
    • Classes like InventoryService that transform raw model output
      into domain-meaningful results.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • FastAPI-specific code (APIRouter, Request, Response, HTTPException)
      — services should be framework-agnostic and unit-testable
      without spinning up FastAPI at all.
    • Model loading/inference code — that's backend/repositories/
      and src/inference/.
"""
