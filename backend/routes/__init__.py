"""
backend.routes
==============

RESPONSIBILITY
--------------
Marks `backend/routes/` as a package and (optionally, once you've
implemented the individual route modules) re-exports each
APIRouter instance so backend/main.py can import them cleanly.

WHAT BELONGS HERE
------------------
    • Nothing beyond imports/re-exports, once implemented.

TODO
----
    - [ ] Once detect.py, report.py, and health.py each define a
          module-level `router = APIRouter(...)`, consider
          re-exporting them here, e.g.:

              from backend.routes.detect import router as detect_router
              from backend.routes.report import router as report_router
              from backend.routes.health import router as health_router

          This is optional polish — backend/main.py can also import
          directly from each submodule without this file doing
          anything. Do whichever is more readable once all three
          routers exist.
"""
