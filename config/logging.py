import logging
import os
from datetime import datetime

def get_logger(name=__name__):
    # Ensure log directory exists
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Timestamped log file
    log_file = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d}.log")

    # Common format
    log_format = "%(asctime)-29s%(levelname)-10s%(name)-40s%(message)s"

    # Configure root logger only once
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),  # keeps console output
        ],
    )

    return logging.getLogger(name)