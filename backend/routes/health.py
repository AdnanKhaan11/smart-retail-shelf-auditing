"""
backend.routes.health

RESPONSIBILITY
--------------
Exposes a liveness/readiness endpoint so orchestration tools
(Docker healthchecks, load balancers, your own frontend) can check
whether the API process is up AND whether the detection model has
finished loading — those are two different questions, and conflating
them is a common production bug (a process can be "alive" but not
yet able to serve real predictions).

PURPOSE
-------
This is deliberately the ONE fully-implemented route in this
project. It is trivial enough that re-deriving it teaches you
nothing, and having one complete, working example route gives you a
concrete pattern to copy when you implement detect.py and report.py.

ARCHITECTURE NOTES
-------------------
    • This route must never depend on the model being loaded to
      respond — if it did, you couldn't distinguish "process crashed"
      from "model still loading" from the outside.
    • model_loaded is read from the shared ModelRepository singleton
      (backend/repositories/model_repository.py), not recomputed here.

ASCII FLOW DIAGRAM
-------------------
    GET /health
        |
        v
    ModelRepository.is_loaded (property read, no heavy work)
        |
        v
    HealthResponse(status="ok", model_loaded=<bool>)

BEST PRACTICES
---------------
    - Keep this endpoint fast and dependency-light on purpose —
      Docker/orchestrators may call it every few seconds.

LEARNING NOTES
--------------
Notice the pattern used here: route function -> Pydantic response
model -> dependency injection via FastAPI's Depends(). You will
repeat this exact three-part pattern in detect.py and report.py,
just with real business logic swapped in for the trivial health
check.

REFERENCES
----------
    - https://fastapi.tiangolo.com/tutorial/dependencies/
"""

from fastapi import APIRouter, Depends

from backend.repositories.model_repository import ModelRepository, get_model_repository
from backend.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    repository: ModelRepository = Depends(get_model_repository),
) -> HealthResponse:
    """Report whether the API process is alive and the model is ready.

    Args:
        repository: Injected ModelRepository singleton (see
            backend/repositories/model_repository.py). FastAPI
            resolves this automatically via Depends().

    Returns:
        HealthResponse with status="ok" (the process responded at
        all) and model_loaded reflecting whether inference is
        actually possible right now.
    """
    return HealthResponse(status="ok", model_loaded=repository.is_loaded)
