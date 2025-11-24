# Change: Migrate logging from stdlib to structlog

## Why

The current logging implementation uses Python's standard library logger with simple string messages. While functional, this provides limited observability and makes debugging production issues difficult. Structlog provides structured logging with contextual data, making logs machine-parseable, easier to search/filter, and significantly more useful for debugging, monitoring, and analytics. It offers better developer experience with context binding, consistent formatting, and seamless integration with modern monitoring tools.

## What Changes

- Replace `logging.getLogger()` calls with `structlog.get_logger()` throughout the codebase
- Create centralized logging configuration module (`src/edm/logging.py`)
- Convert string formatting log messages to structured key-value pairs
- Add context binding for tracking operations (file analysis, API calls, etc.)
- Configure dual-mode output: human-readable colored logs for development, JSON for production
- Add support for log levels via environment variables and CLI flags
- Update all modules to use structured logging patterns
- Add structlog to project dependencies
- Document structured logging conventions and best practices

## Impact

- **Affected specs**: logging (new capability)
- **Affected code**:
  - All Python files in `src/edm/` (logging calls)
  - All Python files in `src/cli/` (logging calls)
  - `pyproject.toml` (add structlog dependency)
  - New file: `src/edm/logging.py` (configuration)
- **Developer experience**: Better debugging, contextual information, easier production troubleshooting
- **Observability**: Machine-parseable logs, integration with monitoring tools, better metrics
- **Performance**: Minimal overhead, lazy evaluation of log messages
- **No breaking changes**: Internal implementation detail, no API changes
