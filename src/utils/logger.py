"""
src.utils.logger

RESPONSIBILITY
--------------
Provides one consistent, pre-configured logger for the whole
project, so that src/training/train.py, backend/main.py, and any
notebook all produce log output in the same format instead of each
reinventing (or forgetting) logging setup.

PURPOSE
-------
Consistent logging is what lets you actually debug a Colab training
run gone wrong, or a FastAPI request that failed in Docker, by
reading timestamped, leveled log output instead of scattered print()
statements.

FULLY IMPLEMENTED
------------------
This entire file is intentionally complete — see this package's
__init__.py for why: there is one standard, correct way to configure
Python's built-in `logging` module for a small project like this,
and reinventing it provides no learning value. Use get_logger(name)
everywhere you would otherwise reach for print().

HOW TO USE
----------
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training run with config: %s", config)
    logger.warning("Validation loss did not improve this epoch")
    logger.error("Failed to load checkpoint from %s", path)

COMMON MISTAKES (worth knowing even though this file is done for you)
------------------------------------------------------------------------
    - Calling logging.basicConfig() in multiple places (e.g. once in
      train.py AND once in main.py) — this can cause duplicate log
      handlers and doubled output. get_logger() below guards against
      this by checking if the root logger already has handlers
      before adding another.
    - Using print() instead of logger calls in any file you write —
      once this module exists, there's no good reason to use print()
      for anything other than genuinely user-facing CLI output
      (e.g. folder_structure.py's summary, which is a one-off script
      output, not application logging).

REFERENCES
----------
    - https://docs.python.org/3/library/logging.html
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a consistently configured logger for the given module name.

    Fully implemented. Safe to call repeatedly (e.g. once per
    module via `get_logger(__name__)`) without producing duplicate
    log handlers.

    Args:
        name: Logger name, conventionally the calling module's
            __name__.
        level: Minimum log level to emit (default: logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
