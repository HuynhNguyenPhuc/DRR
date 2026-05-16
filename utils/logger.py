"""Logging utilities."""

import logging


def setup_logger() -> None:
    """Configure the root logger for the application."""
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import
    if root.handlers:
        root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Suppress noisy third-party loggers
    for lib in ["matplotlib", "PIL", "urllib3", "httpx"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger instance.

    Args:
        name (str): Logger name, typically ``__name__`` of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
