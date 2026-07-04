"""
src.utils
==========

RESPONSIBILITY
--------------
Marks `src/utils/` as a package of small, dependency-light helpers
(logger.py, visualization.py, io.py) reused across the rest of
src/, backend/, and notebooks/.

WHAT BELONGS HERE
------------------
    • Genuinely generic helpers with no project-specific business
      logic — logging setup, file I/O, plotting utilities.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Anything that knows about RT-DETR, SKU-110K, or inventory
      logic specifically — if a function is specific to this
      project's domain, it belongs in src/data/, src/training/,
      src/eval/, src/inference/, or backend/services/, not here.
    • A "misc" dumping ground — if you find yourself adding a
      function here because you "didn't know where else to put it,"
      that's usually a sign it actually belongs in a more specific
      module and you just haven't identified which one yet.
"""

__all__: list[str] = []
