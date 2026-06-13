"""Utility helpers for logging and instrumentation.

This module provides a wrapper to obtain a module-scoped logger that plays
nicely with tqdm progress bars. It installs a lightweight handler that routes
log messages through `tqdm.write(...)` so progress bars are not corrupted by
concurrent logging output.
"""

import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes messages via tqdm.write.

    This avoids interleaving logger output with active tqdm progress bars.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def getLogger(name: str = __name__) -> logging.Logger:
    """Return a configured logger that uses `tqdm.write` for output.

    Parameters
    ----------
    name : str
        Logger name (defaults to the calling module's __name__).
    """
    logger = logging.getLogger(name)

    # Add the handler only once per logger to avoid duplicated messages
    if not any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers):
        handler = TqdmLoggingHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent propagation to the root logger to avoid duplicate output
        logger.propagate = False

    return logger
