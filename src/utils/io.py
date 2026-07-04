"""
src.utils.io

RESPONSIBILITY
--------------
Small, shared file I/O helpers used across the project: reading and
writing JSON (manifests, experiment logs), loading an image file
into a consistent (RGB, PIL.Image) format, and ensuring a directory
exists before writing into it.

FULLY IMPLEMENTED
------------------
Like logger.py, this file is intentionally complete — see this
package's __init__.py for the reasoning. json.load/json.dump,
PIL.Image.open, and Path.mkdir(parents=True, exist_ok=True) each
have one standard, correct way to be used; wrapping them here exists
purely to give the rest of the project ONE consistent, safe entry
point (with clearer error messages) rather than scattering raw
open()/json.load() calls with inconsistent error handling throughout
src/ and backend/.

HOW TO USE
----------
    from src.utils.io import load_json, save_json, load_image, ensure_dir

    manifest = load_json("data/splits/train.json")
    save_json(results, "experiments/latest_results.json")
    image = load_image("data/raw/example.jpg")
    ensure_dir("models/checkpoints/run_003")

WHY THESE FUNCTIONS EXIST INSTEAD OF INLINE CALLS
-----------------------------------------------------
    - load_json/save_json give you one place to add consistent
      error messages (e.g. "manifest file not found at <path> — did
      you complete docs/SDP.md Day 6?") instead of a bare
      FileNotFoundError with no project context, repeated
      differently in every file that happens to load a manifest.
    - load_image always returns a "RGB"-mode PIL.Image — some source
      images are RGBA or grayscale, and silently mixing image modes
      through your pipeline is a classic subtle bug source.
    - ensure_dir centralizes the "parents=True, exist_ok=True"
      pattern so nobody has to remember both flags every time.

REFERENCES
----------
    - https://docs.python.org/3/library/json.html
    - https://pillow.readthedocs.io/en/stable/reference/Image.html
"""

import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    """Load and parse a JSON file.

    Fully implemented.

    Args:
        path: Path to a JSON file.

    Returns:
        The parsed JSON content (typically a dict or list).

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Serialize data to a JSON file, creating parent directories if needed.

    Fully implemented.

    Args:
        data: JSON-serializable data (dict, list, etc.).
        path: Destination path to write to.

    Returns:
        None.
    """
    file_path = Path(path)
    ensure_dir(str(file_path.parent))
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_image(path: str) -> Any:
    """Load an image file as a consistent RGB-mode PIL.Image.

    Fully implemented.

    Args:
        path: Path to an image file.

    Returns:
        A PIL.Image object in "RGB" mode, regardless of the source
        file's original mode (RGBA, grayscale, etc.).

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    from PIL import Image

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found at: {file_path}")
    return Image.open(file_path).convert("RGB")


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, creating parent directories as needed.

    Fully implemented.

    Args:
        path: Directory path to create if it doesn't already exist.

    Returns:
        The Path object for the (now guaranteed to exist) directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
