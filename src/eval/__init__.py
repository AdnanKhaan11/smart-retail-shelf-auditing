"""
src.eval
=========

RESPONSIBILITY
--------------
Marks `src/eval/` as a package containing detection metric
computation, shared by training-time evaluation
(src/training/train.py's compute_metrics) and standalone
post-training evaluation (docs/SDP.md Day 8).

WHAT BELONGS HERE
------------------
    • Metric computation logic (mAP, mAP@50, mAP@50-95, precision,
      recall, F1) and the prediction/target formatting needed to
      feed torchmetrics correctly.

WHAT SHOULD NEVER BELONG HERE
-------------------------------
    • Model loading (src/training/, src/inference/, backend/repositories/)
    • Dataset loading (src/data/)

LEARNING NOTES
--------------
Implement this package's logic exactly ONCE. If you notice yourself
writing metric-computation code a second time inside
src/training/train.py or a notebook, stop and import from here
instead — metric definitions drifting between "what I used during
training" and "what I reported in my README" is a subtle but
resume-damaging inconsistency if anyone (e.g. an interviewer) asks
you to reproduce your numbers.
"""

__all__: list[str] = []
