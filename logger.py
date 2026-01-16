import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = "app_logger",
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    file_name: str = "app.log",
):
    """
    Creates and returns a logger with both console + rotating file handlers.

    :param name: Name of the logger instance
    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_dir: Directory to store log files
    :param file_name: Log file name
    :return: Configured logger
    """

    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # --------------------------
    # Format for all logs
    # --------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------
    # Console handler
    # --------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # --------------------------
    # File handler (with rotation)
    # --------------------------
    file_path = os.path.join(log_dir, file_name)
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=5               # keep 5 backups
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Add handlers only once
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
