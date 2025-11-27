"""Logging configuration using structlog."""

import logging
import sys
from pathlib import Path

import structlog
from structlog.typing import EventDict, WrappedLogger


def add_log_level_name(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add the log level name to the event dict."""
    if method_name == "debug":
        event_dict["level"] = "DEBUG"
    elif method_name == "info":
        event_dict["level"] = "INFO"
    elif method_name == "warning":
        event_dict["level"] = "WARNING"
    elif method_name == "error":
        event_dict["level"] = "ERROR"
    elif method_name == "critical":
        event_dict["level"] = "CRITICAL"
    return event_dict


def configure_logging(
    level: str = "WARNING",
    json_format: bool = False,
    log_file: Path | None = None,
    no_color: bool = False,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output logs in JSON format. Otherwise, use human-readable format.
        log_file: If provided, also write logs to this file.
        no_color: If True, disable colored output.

    Examples:
        >>> configure_logging(level="DEBUG")
        >>> configure_logging(level="INFO", json_format=True)
        >>> configure_logging(level="INFO", log_file=Path("app.log"))
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stderr,
        force=True,  # Force reconfiguration (Python 3.8+)
    )

    # Suppress noisy third-party loggers
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Common processors for all configurations
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,  # Filter logs by configured level
        structlog.stdlib.add_logger_name,
        add_log_level_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add exception processor that formats stack traces
    if json_format:
        processors.append(structlog.processors.format_exc_info)
    else:
        processors.append(structlog.dev.set_exc_info)

    # Configure output format
    if json_format:
        # JSON output for production/structured logging
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        # Human-readable console output with colors (development mode)
        if no_color:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a separate file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        # Create a formatter that processes through structlog
        class StructlogFormatter(logging.Formatter):
            def __init__(self, processors):
                super().__init__()
                self.processors = processors

            def format(self, record):
                # Build event dict from log record
                event_dict = {
                    "event": record.getMessage(),
                    "logger": record.name,
                    "level": record.levelname,
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
                }

                # Add exception info if present
                if record.exc_info:
                    event_dict["exc_info"] = record.exc_info

                # Process through structlog processors
                for processor in self.processors:
                    if callable(processor):
                        try:
                            event_dict = processor(None, "", event_dict)
                        except Exception:
                            pass

                # If it's already a string (from JSONRenderer), return it
                if isinstance(event_dict, (str, bytes)):
                    return event_dict

                # Otherwise, format as JSON
                import json

                return json.dumps(event_dict)

        # File logs always use JSON format for structured output
        file_handler.setFormatter(StructlogFormatter([structlog.processors.JSONRenderer()]))

        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger.

    Args:
        name: Logger name. If not provided, uses the caller's module name.

    Returns:
        Configured structlog logger.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("processing started", file_path="track.mp3")
        >>> logger.error("analysis failed", error="Invalid file format")
    """
    return structlog.get_logger(name)
