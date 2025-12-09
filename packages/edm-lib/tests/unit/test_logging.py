"""Tests for logging configuration and filtering."""

import logging
from io import StringIO

import pytest
import structlog
from edm.logging import configure_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    # Store original state
    original_level = logging.root.level
    original_handlers = logging.root.handlers[:]

    yield

    # Reset logging to original state
    logging.root.setLevel(original_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for handler in original_handlers:
        logging.root.addHandler(handler)

    # Reset structlog
    structlog.reset_defaults()


def test_configure_logging_default_level():
    """Test that default log level is WARNING."""
    # Capture logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    # Configure logging
    configure_logging()

    # Get logger and add handler to capture output
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # Get structlog logger
    logger = structlog.get_logger(__name__)

    # DEBUG and INFO should be filtered out
    logger.debug("debug message")
    logger.info("info message")

    # WARNING and above should appear
    logger.warning("warning message")
    logger.error("error message")

    output = log_stream.getvalue()

    # Check that DEBUG and INFO are not in output
    assert "debug message" not in output
    assert "info message" not in output

    # Check that WARNING and ERROR are in output
    assert "warning message" in output
    assert "error message" in output

    root_logger.removeHandler(handler)


def test_configure_logging_info_level():
    """Test that INFO level shows INFO but not DEBUG."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    configure_logging(level="INFO")

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger = structlog.get_logger(__name__)

    # DEBUG should be filtered out
    logger.debug("debug message")

    # INFO and above should appear
    logger.info("info message")
    logger.warning("warning message")

    output = log_stream.getvalue()

    assert "debug message" not in output
    assert "info message" in output
    assert "warning message" in output

    root_logger.removeHandler(handler)


def test_configure_logging_debug_level():
    """Test that DEBUG level shows all messages."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    configure_logging(level="DEBUG")

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger = structlog.get_logger(__name__)

    # All levels should appear
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")

    output = log_stream.getvalue()

    assert "debug message" in output
    assert "info message" in output
    assert "warning message" in output
    assert "error message" in output

    root_logger.removeHandler(handler)


def test_configure_logging_error_level():
    """Test that ERROR level filters out everything below ERROR."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    configure_logging(level="ERROR")

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger = structlog.get_logger(__name__)

    # DEBUG, INFO, WARNING should be filtered out
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")

    # ERROR and above should appear
    logger.error("error message")
    logger.critical("critical message")

    output = log_stream.getvalue()

    assert "debug message" not in output
    assert "info message" not in output
    assert "warning message" not in output
    assert "error message" in output
    assert "critical message" in output

    root_logger.removeHandler(handler)


def test_configure_logging_invalid_level_defaults_to_warning():
    """Test that invalid log level defaults to WARNING."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    configure_logging(level="INVALID")

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger = structlog.get_logger(__name__)

    # INFO should be filtered (default WARNING)
    logger.info("info message")

    # WARNING should appear
    logger.warning("warning message")

    output = log_stream.getvalue()

    assert "info message" not in output
    assert "warning message" in output

    root_logger.removeHandler(handler)
