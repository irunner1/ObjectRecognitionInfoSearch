"""Logger."""

import logging
import sys


def configure_logger(logger_name: str, log_level=logging.INFO):
    """Create logger.

    Args:
        logger_name: name for logger. Use __name__
        log_level: logging level. Defaults to logging.INFO.
    """

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s\t- %(name)s - %(message)s",
        datefmt="%Y.%m.%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(logger_name)
