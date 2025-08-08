import logging
import sys

def setup_logging():
    # Create a top-level logger for your app
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    # One handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # A simple formatter: you can customize timestamps, etc.
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)

    # Attach the handler (avoid duplicates)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    # Optional: propagate to root if you want others to catch it
    logger.propagate = False

    return logger
