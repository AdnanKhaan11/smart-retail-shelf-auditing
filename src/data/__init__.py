"""
src.data
=========

RESPONSIBILITY
--------------
Marks `src/data/` as a package containing everything needed to turn
the raw SKU-110K annotations into model-ready batches: dataset
loading (dataset.py), image/box preprocessing (preprocessing.py),
augmentation (augmentations.py), and batching (dataloader.py).

WHAT BELONGS HERE
------------------
    • Anything that transforms raw data on disk into tensors a model
      can consume.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Model definition or training loop code (src/training/)
    • Metric computation (src/eval/)
    • Anything backend/frontend-specific

LEARNING NOTES
--------------
Every function in this package should be usable from THREE different
contexts without modification: a Colab training run
(src/training/train.py), a local EDA/error-analysis notebook
(notebooks/), and (for preprocessing specifically) potentially
inference too. If a function only works inside train.py, it's
probably too tightly coupled to Trainer-specific assumptions.
"""

__all__: list[str] = []
