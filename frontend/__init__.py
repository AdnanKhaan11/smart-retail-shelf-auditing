"""
frontend
=========

RESPONSIBILITY
--------------
Marks `frontend/` as an importable Python package containing the
Gradio demo UI for the Smart Retail Shelf Auditing System.

WHAT BELONGS HERE
------------------
    • UI layout and calls to the backend API.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Model loading or inference logic — this package must call the
      backend's HTTP API (POST /detect, POST /report), never import
      src/inference/inference.py or backend/repositories/ directly.
      Duplicating inference logic here would mean two different code
      paths could silently drift out of sync, and would prevent you
      from ever deploying the frontend and backend as separate
      services (e.g. frontend on Hugging Face Spaces, backend
      elsewhere) later.
    • Business logic (counting, thresholds) — that's
      backend/services/inventory.py's job; the frontend only
      displays what the backend already computed.
"""

__all__: list[str] = []
