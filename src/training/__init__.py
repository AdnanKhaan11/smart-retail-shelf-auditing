"""
src.training
=============

RESPONSIBILITY
--------------
Marks `src/training/` as an importable Python package containing the
RT-DETR fine-tuning pipeline: configuration (config.py) and the
orchestration script that actually runs training (train.py).

WHAT BELONGS HERE
------------------
    • Training orchestration and configuration ONLY.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Dataset loading/preprocessing logic (src/data/)
    • Evaluation metric computation (src/eval/)
    • Anything FastAPI/backend-related — this package must be
      importable and runnable on Google Colab, with zero dependency
      on backend/ (see docs/SDP.md Section 5 for why training and
      serving environments are kept separate).

LEARNING NOTES
--------------
This package is meant to be run on Colab (python -m src.training.train
or a notebook cell that calls src.training.train.main()), while
backend/ is meant to be run locally/in Docker. If you ever find
yourself importing something from backend/ into this package, stop —
that's a sign the separation of concerns has broken down.
"""

__all__: list[str] = []
