"""
logger_config.py
Sets up a named logger that writes to a timestamped file and console.
"""
import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = "logs", name: str = "FlightProject", level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    # File handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logger initialized, writing to {log_filename}")
    return logger

# Convenience
logger = setup_logger()
