# Implementation Tasks

## 1. Add Dependencies

- [x] Add structlog to `pyproject.toml` dependencies
- [x] Run dependency installation to verify compatibility

## 2. Create Logging Configuration Module

- [x] Create `src/edm/logging.py` module
- [x] Implement `configure_logging()` function with level, format, and output options
- [x] Configure processors for development mode (colored console output)
- [x] Configure processors for production mode (JSON output)
- [x] Add timestamp formatting (ISO 8601)
- [x] Add log level rendering
- [x] Add logger name rendering
- [x] Add exception formatting with stack traces
- [x] Support context variables for request/operation tracking
- [x] Add file output option for persistent logs

## 3. Update Core Library Modules

- [x] Update `src/edm/analysis/bpm_detector.py`:
  - Replace `logging.getLogger()` with `structlog.get_logger()`
  - Convert log messages to structured format with key-value pairs
  - Add context binding for file paths, methods, timing
- [x] Update `src/edm/analysis/bpm.py`:
  - Migrate to structlog
  - Add structured context for BPM analysis operations
- [x] Update `src/edm/analysis/structure.py`:
  - Migrate to structlog
  - Add context for structure analysis
- [x] Update `src/edm/external/spotify.py`:
  - Migrate to structlog
  - Add context for API operations (track searches, requests)
  - Log request/response metadata
- [x] Update `src/edm/external/beatport.py`:
  - Migrate to structlog
  - Add API operation context
- [x] Update `src/edm/external/tunebat.py`:
  - Migrate to structlog
  - Add API operation context
- [x] Update `src/edm/io/metadata.py`:
  - Migrate to structlog
  - Add context for metadata operations
- [x] Update `src/edm/config.py`:
  - Migrate to structlog
  - Add context for configuration loading

## 4. Update CLI Modules

- [x] Update `src/cli/main.py`:
  - Initialize structlog configuration on startup
  - Add CLI flag for log level (--log-level DEBUG/INFO/WARNING/ERROR)
  - Add --verbose flag that maps to --log-level DEBUG
  - Add CLI flag for JSON log output (--json-logs)
  - Add CLI flag for log file path (--log-file PATH)
- [x] Update `src/cli/commands/analyze.py`:
  - Migrate to structlog
  - Add context binding for analyze operations
- [x] Update `src/cli/commands/evaluate.py`:
  - Migrate to structlog

## 5. Testing

- [x] Verify all existing functionality still works correctly
- [x] Test CLI flags for log configuration

## 6. Cleanup

- [x] Remove stdlib logging from `src/edm/external/beatport.py`
- [x] Remove stdlib logging from `src/edm/external/tunebat.py`
- [x] Verify no `print()` calls used where logging should be
