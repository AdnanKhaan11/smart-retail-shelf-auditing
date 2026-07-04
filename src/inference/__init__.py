"""
src.inference
===============

RESPONSIBILITY
--------------
Marks `src/inference/` as a package containing the framework-agnostic
inference logic reused by backend/routes/detect.py and
backend/routes/report.py.

WHAT BELONGS HERE
------------------
    • A pure function that takes a loaded model + processor + one
      image and returns detections — nothing more.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • FastAPI imports (APIRouter, UploadFile, HTTPException, etc.) —
      see backend/services/inventory.py's module docstring for the
      same "framework-agnostic" principle applied there; the exact
      same reasoning applies here.
    • Model LOADING/lifecycle logic — that is
      backend/repositories/model_repository.py's job. This package
      only knows how to USE an already-loaded model, not how to load
      one.

LEARNING NOTES
--------------
A good test of whether this package is properly decoupled: you
should be able to write a standalone test or CLI script that loads a
model directly with plain `transformers` calls (no FastAPI, no
backend/ import at all) and calls run_inference() on it successfully.
If you can't, something here has become too tightly coupled to the
backend.
"""

__all__: list[str] = []
