"""
Centralized logging configuration for the MAAP Agent Builder.
"""

import contextlib
import json
import logging
import os
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional, Union

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """

    def __init__(self, **kwargs):
        self.json_ensure_ascii = kwargs.pop("json_ensure_ascii", False)
        super().__init__(**kwargs)

    def format(self, record):
        logobj = {}
        logobj["timestamp"] = self.formatTime(record)
        logobj["name"] = record.name
        logobj["level"] = record.levelname
        logobj["message"] = record.getMessage()

        if record.exc_info:
            logobj["exc_info"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            logobj["extra"] = record.extra

        if hasattr(record, "stack_info") and record.stack_info:
            logobj["stack_info"] = record.stack_info

        return json.dumps(logobj, ensure_ascii=self.json_ensure_ascii)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_str: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    module_log_levels: Optional[dict] = None,
    json_output: bool = False,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level, either an ``int`` (e.g. ``logging.INFO``) or a
            case-insensitive level name (e.g. ``"INFO"``).
        format_str: Log format string
        log_file: Optional path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        module_log_levels: Dictionary mapping module names to specific log levels
        json_output: Whether to output logs in JSON format
    """
    # Accept either a numeric level or a string level name.
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
    else:
        numeric_level = level
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Create formatter
    if json_output:
        formatter = JsonFormatter(json_ensure_ascii=False, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handlers = []

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=numeric_level, format=format_str, handlers=handlers)

    # Set specific log levels for modules if provided
    if module_log_levels:
        for module, mod_level in module_log_levels.items():
            module_level = getattr(logging, mod_level.upper(), None)
            if isinstance(module_level, int):
                logging.getLogger(module).setLevel(module_level)

    # Suppress excessive logging from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger, typically __name__

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)

    # Configure default logging level from environment if not already configured
    if not logging.getLogger().handlers:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        # Pass the string log level to configure_logging, not the numeric level
        configure_logging(level=log_level)

    return logger


class ContextLogger:
    """
    Logger wrapper that adds context to log messages.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}

    def add_context(self, **kwargs):
        """Add context data that will be included with every log message."""
        self.context.update(kwargs)
        return self

    def remove_context(self, *keys):
        """Remove specific context data by key."""
        for key in keys:
            self.context.pop(key, None)
        return self

    def clear_context(self):
        """Clear all context data."""
        self.context.clear()
        return self

    def _log_with_context(self, level, msg, *args, **kwargs):
        """Add the stored context to the log record."""
        extra = kwargs.get("extra", {})
        if hasattr(extra, "update"):
            extra.update(self.context)
        else:
            extra = self.context.copy()
        kwargs["extra"] = extra

        if args and isinstance(args[-1], dict) and len(args) > 1:
            # If the last arg is a dict and there are other args,
            # assume it's meant to be the context
            context_dict = args[-1]
            args = args[:-1]
            if hasattr(extra, "update"):
                extra.update(context_dict)

        return getattr(self.logger, level)(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        return self._log_with_context("debug", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self._log_with_context("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._log_with_context("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self._log_with_context("error", msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self._log_with_context("critical", msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        return self._log_with_context("exception", msg, *args, **kwargs)


@contextlib.contextmanager
def log_timed_operation(logger, operation_name, log_level="info"):
    """
    Context manager to log the execution time of an operation.

    Args:
        logger: The logger instance to use
        operation_name: Name of the operation being timed
        log_level: Log level to use for the messages ('debug', 'info', etc.)

    Example:
        ```python
        logger = get_context_logger(__name__)
        with log_timed_operation(logger, "data processing"):
            process_data()
        ```
    """
    start_time = time.time()
    try:
        logger._log_with_context(log_level, f"Starting {operation_name}")
        yield
    finally:
        elapsed = time.time() - start_time
        logger._log_with_context(
            log_level, f"Completed {operation_name} in {elapsed:.3f} seconds"
        )


def setup_exception_logging():
    """
    Set up global exception hooks to ensure all unhandled exceptions are logged.

    This should be called once at application startup.
    """
    logger = get_logger("exception_handler")

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt exceptions
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    # Set the exception hook
    sys.excepthook = handle_exception


def get_context_logger(name: str) -> ContextLogger:
    """
    Get a context-aware logger with the specified name.

    Args:
        name: Name of the logger, typically __name__

    Returns:
        A context-aware logger instance
    """
    return ContextLogger(get_logger(name))


def sanitize_connection_string(conn_str: str) -> str:
    """Return a safe-for-logging version of a MongoDB connection string.

    Strips credentials and replaces them with ``***`` so the URI can
    still be identified (host / database) without exposing secrets.
    """
    return re.sub(
        r"://([^:]+):([^@]+)@", r"://\1:***@", str(conn_str)
    )
