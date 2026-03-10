from __future__ import annotations

import logging
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "chemreport"


def get_log_file_path() -> Path:
    resolved = getattr(configure_logging, "_resolved_path", None)
    if resolved is not None:
        return resolved

    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path.home() / ".chemreport_mvp" / "logs")
    else:
        candidates.append(Path(__file__).resolve().parent.parent / "outputs" / "logs")
    candidates.append(Path(tempfile.gettempdir()) / "chemreport_mvp" / "logs")

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            log_path = base / "chemreport.log"
            with open(log_path, "a", encoding="utf-8"):
                pass
            configure_logging._resolved_path = log_path
            return log_path
        except OSError:
            continue

    fallback = candidates[-1] / "chemreport.log"
    configure_logging._resolved_path = fallback
    return fallback


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(configure_logging, "_configured", False):
        logger.setLevel(level)
        return logger

    log_path = get_log_file_path()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    logger.setLevel(level)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    try:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Failed to open log file, using console only: %s", log_path)

    configure_logging._configured = True
    logger.info("Logging configured. File: %s", log_path)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
