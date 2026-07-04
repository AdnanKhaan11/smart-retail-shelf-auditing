"""
src.utils.visualization

RESPONSIBILITY
--------------
Offline, analysis-oriented plotting utilities — NOT the same thing
as src/inference/inference.py's draw_boxes_on_image(), which draws
boxes on ONE image at real-request time. This module is for
notebooks: exploring dataset statistics (docs/SDP.md Day 3 EDA) and
visually comparing predictions against ground truth across MANY
images at once (docs/SDP.md Day 10 error analysis).

PURPOSE
-------
Day 3's explicit EDA deliverable ("average of 147 boxes per image,
median box area X px^2" — see docs/SDP.md) and Day 10's error
analysis deliverable both need real plots, not just printed numbers,
to actually be useful documents. This module is where that plotting
code lives so notebooks/01_eda.ipynb and notebooks/02_error_analysis.ipynb
stay thin (call a function from here, don't inline forty lines of
matplotlib per cell).

ARCHITECTURE NOTES
-------------------
    • Both functions below take already-loaded, in-memory data
      (a dataset object, or lists of images/predictions/ground
      truths) — this module has no knowledge of file paths, manifest
      formats, or how the data was produced. That's src/data/'s job.
    • Deliberately separate from src/inference/inference.py's
      draw_boxes_on_image(): that function is a hot-path, single-image,
      request-time utility; these are cold-path, exploratory,
      multi-image, notebook-only utilities. Don't merge them just
      because both "draw boxes" — their performance and API
      requirements are genuinely different.

ASCII FLOW DIAGRAM
-------------------
    (Day 3) datasets.Dataset (train split)
            |
            v
    plot_bbox_distribution(dataset)   <- YOU implement this
            |
            v
    matplotlib histograms: boxes-per-image, box-area distribution

    (Day 10) list[image], list[predictions], list[ground_truths]
            |
            v
    plot_predictions_grid(images, predictions, ground_truths)   <- YOU implement this
            |
            v
    a grid of images with predicted boxes (one color) and ground
    truth boxes (another color) overlaid, for visual failure-mode
    inspection

TODO
----
    - [ ] Implement plot_bbox_distribution(dataset):
          1. Compute boxes-per-image counts across the dataset
          2. Compute box area (width * height) for every box
          3. Plot two histograms (matplotlib) side by side or
             stacked: boxes-per-image, and box area distribution
          4. This directly produces the numbers Day 3's EDA
             deliverable in docs/SDP.md asks you to report
    - [ ] Implement plot_predictions_grid(images, predictions,
          ground_truths, n=9):
          1. Select n images (or all, if fewer than n provided)
          2. For each, draw predicted boxes in one color and ground
             truth boxes in a different color on the same image
             (matplotlib patches, or reuse a drawing helper — but
             keep the OUTPUT a matplotlib figure/grid, not a saved
             single image, since the point is COMPARING many at once)
          3. Arrange into a grid (e.g. matplotlib subplots) and
             display/save the figure

HINTS
-----
    - matplotlib.patches.Rectangle is the standard way to overlay a
      box on an image inside a matplotlib Axes — don't reach for
      OpenCV here, since these are notebook-only, exploratory plots,
      not something you need to serve or process at speed.
    - A consistent color convention (e.g. green = ground truth,
      red = prediction) across every use of
      plot_predictions_grid() makes your error-analysis notebook
      far easier to read later — pick one and stick with it.

COMMON MISTAKES
----------------
    - Building a new, slightly different plotting snippet inside
      each notebook cell instead of calling a shared function from
      here — this is exactly the duplication this module exists to
      prevent.

BEST PRACTICES
---------------
    - Save a few of your most informative plot outputs from Day 3
      and Day 10 directly into docs/ (or reference them from your
      README) — a well-chosen box-count histogram or a predictions
      grid with one clear failure case is often more persuasive to a
      reviewer than a paragraph of prose.

LEARNING NOTES
--------------
This file is a good one to genuinely enjoy — it's where your EDA and
error analysis actually become visible, tangible artifacts rather
than numbers you thought about but never showed anyone.

REFERENCES
----------
    - https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
"""

from typing import Any


def plot_bbox_distribution(dataset: Any) -> None:
    """Plot histograms of boxes-per-image and box-area distribution.

    Args:
        dataset: A loaded datasets.Dataset (e.g. the training split
            from src/data/dataset.py's load_split()), with an
            "annotations" field per example.

    Returns:
        None. Displays (or saves, your choice once implemented) a
        matplotlib figure with the two histograms described in this
        file's module docstring.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("plot_bbox_distribution() is not implemented yet")


def plot_predictions_grid(
    images: list[Any],
    predictions: list[list[dict]],
    ground_truths: list[list[dict]],
    n: int = 9,
) -> None:
    """Plot a grid comparing predicted boxes against ground truth boxes.

    Args:
        images: A list of PIL.Image objects.
        predictions: A list of per-image prediction lists (same
            shape as src/inference/inference.py's run_inference()
            output), one entry per image in `images`.
        ground_truths: A list of per-image ground-truth box lists,
            one entry per image in `images`.
        n: Maximum number of images to include in the grid.

    Returns:
        None. Displays (or saves) a matplotlib figure grid, useful
        for the Day 10 error-analysis deliverable in docs/SDP.md.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("plot_predictions_grid() is not implemented yet")
