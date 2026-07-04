"""
frontend.app

RESPONSIBILITY
--------------
The Gradio demo application: an image upload widget, a display of
the annotated detection image, and an inventory report panel
(detected count, stock-level percentage, low-stock warning). This
file ONLY calls the backend's REST API — it must never import
model/inference code directly (see this package's __init__.py for
why).

PURPOSE
-------
docs/SDP.md Day 13's explicit deliverable, and arguably the single
most-viewed artifact on your resume — most people evaluating your
GitHub will click a live demo link before reading a single line of
your code. Make this feel like a small, polished product, not a bare
Gradio default layout.

ARCHITECTURE NOTES
-------------------
    • This file calls the backend over HTTP (via the `requests`
      library) at a configurable BACKEND_URL, exactly the same way
      any external client would — this keeps frontend and backend
      genuinely decoupled and independently deployable/testable
      (see docs/SDP.md Section 12's deployment pipeline, where
      frontend and backend are explicitly separate Docker
      containers).
    • Keep the actual Gradio interface definition (gr.Blocks / the
      launched app) at module level so `python -m frontend.app` runs
      it directly, matching the pattern in backend/main.py (module
      exposes a ready-to-run object at import time).

ASCII FLOW DIAGRAM
-------------------
    User uploads image via Gradio widget
            |
            v
    on_submit(image)              <- YOU implement this
            |
            |  1. encode image, POST to {BACKEND_URL}/report
            |  2. parse JSON response (InventoryReportResponse shape)
            |  3. return (annotated_image, report_text) for display
            v
    Gradio updates: annotated image panel + report text panel

TODO
----
    - [ ] Implement on_submit(image):
          1. Save/serialize the uploaded image (Gradio gives you a
             PIL.Image or numpy array depending on component config)
             into a form `requests.post(..., files={"file": ...})`
             can send
          2. POST to f"{BACKEND_URL}/report" (optionally accept a
             baseline_count from a Gradio number input as a query
             param, matching backend/routes/report.py's signature)
          3. Handle non-200 responses gracefully — show an error
             message in the UI rather than letting an unhandled
             exception crash the Gradio callback
          4. Parse the JSON response and extract whatever the
             backend returns (annotated image if you chose to return
             one from the API, plus the structured report fields)
          5. Return values matching your gr.Interface/gr.Blocks
             output components (see Hints below)
    - [ ] Build the actual UI layout using gr.Blocks (recommended
          over the simpler gr.Interface, since you want multiple
          coordinated outputs — annotated image + a formatted report
          — laid out deliberately, not auto-generated)
    - [ ] Wire on_submit to your submit button's .click() handler

HINTS
-----
    - Use `gr.Blocks()` with an explicit layout (gr.Row/gr.Column) so
      you control exactly where the upload widget, submit button,
      annotated image, and report panel sit — the default
      auto-layout from gr.Interface tends to look generic.
    - Read BACKEND_URL from an environment variable
      (os.environ.get("BACKEND_URL", "http://localhost:8000")) so
      the same code works whether the backend is running locally or
      inside a separate Docker container (docker-compose sets this
      via service networking).
    - Consider a gr.Markdown component for the report panel so you
      can format the stock-level percentage and low-stock warning
      with basic styling (bold, color-coded text) rather than plain
      text.

COMMON MISTAKES
----------------
    - Importing anything from backend/ or src/ directly instead of
      calling the API over HTTP — this defeats the entire point of
      having a separate, independently deployable frontend.
    - Not handling backend connection errors (e.g. backend not yet
      started) — show a clear "backend unavailable" message instead
      of letting Gradio display a raw traceback to a demo viewer.

BEST PRACTICES
---------------
    - Add a short markdown description at the top of the Gradio
      Blocks layout explaining what the app does in one sentence —
      remember someone clicking your demo link may have zero context
      beyond your README.
    - Include a couple of sample/example images (gr.Examples) if you
      have any non-sensitive sample shelf photos, so a visitor can
      try the demo with one click instead of needing their own image.

LEARNING NOTES
--------------
This file is a good place to feel the payoff of everything else
you've built — once backend/ is fully implemented and running, this
file is a relatively small amount of code that turns your whole
pipeline into something a non-technical person (a recruiter!) can
actually click through and understand in 30 seconds.

REFERENCES
----------
    - https://www.gradio.app/docs/gradio/blocks
    - https://requests.readthedocs.io/en/latest/
"""

import os
from typing import Any

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def on_submit(image: Any, baseline_count: int | None = None) -> tuple[Any, str]:
    """Send an uploaded image to the backend and format the response.

    Args:
        image: The uploaded image from the Gradio input component
            (typically a PIL.Image or numpy array, depending on
            component configuration).
        baseline_count: Optional expected full-shelf product count,
            forwarded to the backend's POST /report endpoint.

    Returns:
        A (annotated_image, report_markdown) tuple: the annotated
        detection image to display, and a formatted markdown string
        summarizing the inventory report (detected count, stock
        level, low-stock warning if applicable).

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("on_submit() is not implemented yet")


def build_interface() -> Any:
    """Construct the Gradio Blocks interface for this demo.

    Returns:
        A gr.Blocks instance, ready to be launched.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    # TODO: import gradio as gr, build a gr.Blocks() layout with:
    #   - an image upload component
    #   - an optional baseline_count number input
    #   - a submit button wired to on_submit()
    #   - an output image component (annotated detections)
    #   - an output markdown/text component (inventory report)
    raise NotImplementedError("build_interface() is not implemented yet")


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
