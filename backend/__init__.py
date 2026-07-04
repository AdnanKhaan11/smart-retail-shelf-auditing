"""
backend
=======

RESPONSIBILITY
--------------
Marks `backend/` as an importable Python package and exposes the
FastAPI application object so it can be run with:

    uvicorn backend.main:app --reload

PURPOSE
-------
This is the serving layer of the Smart Retail Shelf Auditing System.
It is deliberately separate from `src/` (the training/inference
package) — see the "Architecture Notes" in backend/main.py for why
that separation matters.

WHAT BELONGS HERE
------------------
    • Package-level metadata (__version__, __all__)
    • Nothing else — keep this file tiny

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Route definitions (those live in backend/routes/)
    • Business logic (backend/services/)
    • Model loading logic (backend/repositories/)

LEARNING NOTES
--------------
Empty-looking __init__.py files are not "nothing" — they are what
turns a folder into an importable Python package. Resist the urge to
delete them because they "don't do anything."
"""

__all__: list[str] = []

# TODO: set this to match your project's actual versioning scheme
__version__ = "0.1.0"
