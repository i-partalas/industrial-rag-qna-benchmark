import logging

from colorlog import ColoredFormatter


def setup_logger():
    """Sets up a logging system with both file and console handlers
    at different log levels for detailed evaluation of LLMs."""
    logger = logging.getLogger("llm_benchmarking")

    # Prevent configuring the logger multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler for detailed debug logs
    fh = logging.FileHandler("llm_evaluation.log")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    fh.setFormatter(file_formatter)

    # Console handler for user-friendly colored logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    LOGFORMAT = (
        "%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - "
        "%(log_color)s%(message)s%(reset)s"
    )
    color_formatter = ColoredFormatter(LOGFORMAT)
    ch.setFormatter(color_formatter)

    # Adding handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()
