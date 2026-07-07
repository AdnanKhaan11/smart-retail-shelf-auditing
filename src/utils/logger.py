"""
src.utils.logger

Provides one reusable logger for the entire project.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Usually __name__ of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured Logger object.
    """

    # Get existing logger or create a new one
    logger = logging.getLogger(name)

    # Set minimum log level
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.handlers:

        # Print logs to terminal
        handler = logging.StreamHandler(stream=sys.stdout)

        # Log message format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Attach formatter to handler
        handler.setFormatter(formatter)

        # Attach handler to logger
        logger.addHandler(handler)

        # Stop logs from propagating to parent logger
        logger.propagate = False

    return logger
