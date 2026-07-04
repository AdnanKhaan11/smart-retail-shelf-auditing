"""
backend.main

RESPONSIBILITY
--------------
The FastAPI application entrypoint. Creates the `app` object,
registers middleware, wires up routers, and manages the model's
startup/shutdown lifecycle. Run with:

    uvicorn backend.main:app --reload

PURPOSE
-------
This file is the "composition root" of the backend — the one place
where all the independently-built pieces (routes, services,
repositories) get wired together into a single running application.
Everything above this file (schemas, routes, services, repositories)
should be understandable and testable in isolation; this file is
where they finally meet.

ARCHITECTURE NOTES
-------------------
    • WHY src/ and backend/ are separate (see docs/SDP.md Section 5
      for the full reasoning): training code (src/) and serving code
      (backend/) have different lifecycles, different compute needs
      (GPU-heavy training vs. lightweight-ish serving), and should be
      deployable independently. This file is the boundary — it
      imports the trained checkpoint (an artifact), never the
      training code itself.
    • Model loading happens ONCE, at startup, via a lifespan event —
      never inside a request handler (see model_repository.py's
      "Common Mistakes" section for why).

ASCII FLOW DIAGRAM
-------------------
    uvicorn backend.main:app
            |
            v
    create_app()
        |-- instantiate FastAPI(...)
        |-- add CORS middleware
        |-- register lifespan (startup: load model / shutdown: cleanup)
        |-- include_router(detect_router)
        |-- include_router(report_router)
        |-- include_router(health_router)
            v
    app  (ASGI application object uvicorn actually serves)

TODO
----
    - [ ] In the lifespan startup block: construct a ModelRepository
          with the correct model_path (read from
          configs/inference_config.yaml — don't hardcode the path
          here either), call repository.load_model(), and store it
          somewhere backend.repositories.model_repository's
          get_model_repository() can return it from (module-level
          singleton is the simplest correct approach — see the TODO
          in that file).
    - [ ] Decide whether you need a global exception handler (e.g.
          for RuntimeError raised by an unloaded model) that maps to
          a consistent ErrorResponse shape from schemas.py.
    - [ ] Consider adding request logging middleware using
          src/utils/logger.py once that module is implemented.

HINTS
-----
    - FastAPI's `lifespan` context manager (replacing the older
      @app.on_event("startup") style) is the current recommended way
      to run startup/shutdown code — look it up if you haven't used
      it before.
    - CORS is already configured permissively below for local
      development convenience — TIGHTEN this before any real
      deployment (see the TODO comment inline).

COMMON MISTAKES
----------------
    - Loading the model at import time (i.e., as a module-level
      statement outside any function) — this makes the app import
      slow, complicates testing (importing the module for a unit
      test now requires GPU/model access), and doesn't give you a
      clean hook for error handling if loading fails.
    - Forgetting to register a router you've implemented — a route
      file with a fully implemented `router = APIRouter(...)` does
      nothing until it's included here with `app.include_router(...)`.

BEST PRACTICES
---------------
    - Keep this file focused purely on wiring/composition. If you
      find yourself writing business logic or route handlers
      directly in this file, move them to the appropriate
      routes/services module instead.

LEARNING NOTES
--------------
This file is intentionally one of the few in backend/ that is mostly
already implemented — application wiring (instantiate app, add
middleware, include routers) is genuine boilerplate with very few
"right" ways to do it differently, unlike business logic. Spend your
learning time on services/inventory.py and repositories/
model_repository.py instead.

REFERENCES
----------
    - https://fastapi.tiangolo.com/advanced/events/ (lifespan events)
    - https://fastapi.tiangolo.com/tutorial/cors/
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.detect import router as detect_router
from backend.routes.health import router as health_router
from backend.routes.report import router as report_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI while the application runs. Code
        before `yield` runs at startup; code after `yield` runs at
        shutdown.
    """
    # TODO: construct + load the ModelRepository singleton here.
    # Example shape (uncomment and complete once ModelRepository is
    # implemented):
    #
    #     from backend.repositories.model_repository import ModelRepository
    #     repository = ModelRepository(model_path="models/exported/model.onnx")
    #     repository.load_model()
    #     # TODO: make `repository` discoverable by get_model_repository()
    #
    yield
    # TODO: any shutdown/cleanup logic goes here (usually not much
    # needed for a read-only inference model, but document if so).


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        A fully configured FastAPI app instance, ready for uvicorn
        to serve. Router registration happens here so the app object
        returned is complete and testable via FastAPI's TestClient.
    """
    app = FastAPI(
        title="Smart Retail Shelf Auditing System",
        description="Detects and counts retail products on shelf images "
        "to flag low-stock sections for restocking.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # NOTE: this permissive CORS config is fine for local development
    # only. TODO: restrict allow_origins to your actual frontend's
    # origin(s) before any real deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(detect_router)
    app.include_router(report_router)
    app.include_router(health_router)

    return app


app = create_app()
