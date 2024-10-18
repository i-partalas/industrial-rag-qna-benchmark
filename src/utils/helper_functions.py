import os

from utils.logger import logger


def determine_max_concurrency():
    """
    Determines the maximum number of threads (max_concurrency) based on the number
    of logical processors available on the system.

    Returns:
        int: Recommended max_concurrency value based on the number of logical processors.
    """
    # Get the number of logical processors
    logical_processors = os.cpu_count()
    if logical_processors is None:
        # Fallback to a default value if logical_processors could not be determined
        logical_processors = 4

    # Use the number of logical processors as the max_concurrency
    max_concurrency = logical_processors
    logger.debug(f"Max_concurrency based on system resources: {max_concurrency}")

    return max_concurrency
