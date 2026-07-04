"""
backend.repositories
======================

RESPONSIBILITY
--------------
Marks `backend/repositories/` as a package. Repositories abstract
away WHERE and HOW a resource is loaded/stored (here: the trained
model) from the code that USES that resource (routes, services).

WHAT BELONGS HERE
------------------
    • ModelRepository — loads and provides access to the fine-tuned
      RT-DETR model as a long-lived singleton.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Actual inference/forward-pass logic — that belongs in
      src/inference/inference.py. This layer only owns the model's
      lifecycle (load once at startup, provide access, unload on
      shutdown), not what you DO with the model.
"""
