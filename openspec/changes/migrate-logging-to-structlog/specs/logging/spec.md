# Logging Specification

## ADDED Requirements

### Requirement: Structured Logging with Contextual Data

The application SHALL use structlog for all logging operations to provide structured, machine-parseable logs with contextual information.

#### Scenario: Basic structured logging
- **WHEN** code logs an event
- **THEN** the log SHALL be structured as key-value pairs
- **AND** the log SHALL include a descriptive event name
- **AND** the log SHALL include relevant contextual data as separate fields
- **AND** the log SHALL NOT use string interpolation for data

#### Scenario: Context binding for operations
- **WHEN** an operation begins (e.g., analyzing a file)
- **THEN** a logger SHALL be bound with operation context
- **AND** all subsequent logs in that operation SHALL include the bound context
- **AND** context SHALL include identifiers like filepath, filename, operation type
- **AND** nested operations SHALL inherit and extend parent context

#### Scenario: Timing and performance logging
- **WHEN** an operation completes
- **THEN** elapsed time SHALL be logged as a numeric field
- **AND** timing data SHALL be in seconds with millisecond precision
- **AND** timing SHALL enable performance analysis and monitoring
- **AND** slow operations SHALL be identifiable from logs

### Requirement: Dual-Mode Output Configuration

The application SHALL support both human-readable development logging and JSON production logging.

#### Scenario: Development mode output
- **WHEN** the application runs in development mode (default)
- **THEN** logs SHALL be formatted for human readability
- **AND** logs SHALL include colored output for different log levels
- **AND** logs SHALL show timestamps in ISO 8601 format
- **AND** logs SHALL display key-value pairs clearly
- **AND** output SHALL be optimized for terminal viewing

#### Scenario: Production mode JSON output
- **WHEN** JSON logging is enabled (via CLI flag or environment variable)
- **THEN** logs SHALL be output as single-line JSON objects
- **AND** each log line SHALL be valid, parseable JSON
- **AND** JSON SHALL include all context and metadata
- **AND** JSON SHALL be compatible with log aggregation tools
- **AND** JSON SHALL NOT include ANSI color codes

#### Scenario: Log level configuration
- **WHEN** the application starts
- **THEN** log level SHALL be configurable via CLI flag (--log-level)
- **AND** log level SHALL be configurable via environment variable (LOG_LEVEL)
- **AND** supported levels SHALL be DEBUG, INFO, WARNING, ERROR
- **AND** default level SHALL be INFO
- **AND** logs below the configured level SHALL NOT be output

### Requirement: File Logging Support

The application SHALL support writing logs to files with optional rotation.

#### Scenario: Log file output
- **WHEN** a log file path is configured (--log-file flag)
- **THEN** logs SHALL be written to the specified file
- **AND** logs SHALL be written in the configured format (JSON or console)
- **AND** logs SHALL also be written to console (dual output)
- **AND** log file SHALL be created if it doesn't exist

#### Scenario: Log rotation
- **WHEN** file logging is enabled and log file grows large
- **THEN** log rotation SHALL be supported to prevent unbounded growth
- **AND** rotation SHALL be based on file size (default: 10MB)
- **AND** rotated files SHALL be numbered or timestamped
- **AND** old rotated files SHALL be automatically cleaned up

#### Scenario: Log file permissions
- **WHEN** log files are created
- **THEN** appropriate file permissions SHALL be set
- **AND** log directory SHALL be created if it doesn't exist
- **AND** file write errors SHALL be handled gracefully
- **AND** users SHALL be notified if logging to file fails

### Requirement: Consistent Event Naming and Context Fields

The application SHALL follow consistent conventions for event names and context field names.

#### Scenario: Event naming convention
- **WHEN** logging an event
- **THEN** event names SHALL use snake_case format
- **AND** event names SHALL be descriptive and specific
- **AND** completed actions SHALL use past tense (e.g., "file_analyzed")
- **AND** ongoing actions SHALL use present progressive (e.g., "analyzing_file")
- **AND** errors SHALL clearly indicate failure (e.g., "analysis_failed")

#### Scenario: Standard context fields
- **WHEN** logging operations on files
- **THEN** context SHALL include "filepath" (full path) and "filename" (basename)
- **WHEN** logging timed operations
- **THEN** context SHALL include "elapsed_time" in seconds
- **WHEN** logging API operations
- **THEN** context SHALL include "operation", "endpoint", and "status_code"
- **WHEN** logging analysis results
- **THEN** context SHALL include relevant metrics (bpm, confidence, num_beats, etc.)
- **AND** numeric values SHALL be rounded appropriately (1-3 decimal places)

#### Scenario: Error context
- **WHEN** logging errors or exceptions
- **THEN** error message SHALL be in "error" field
- **AND** exception type SHALL be in "error_type" field
- **AND** stack trace SHALL be included for unexpected errors
- **AND** context SHALL include operation state at time of error

### Requirement: Exception and Stack Trace Logging

The application SHALL properly log exceptions with full context and stack traces.

#### Scenario: Exception logging
- **WHEN** an exception occurs
- **THEN** the exception SHALL be logged at ERROR level
- **AND** the exception message SHALL be included in the log
- **AND** the exception type SHALL be identified
- **AND** full stack trace SHALL be included
- **AND** operation context SHALL be preserved

#### Scenario: Caught exceptions
- **WHEN** an exception is caught and handled
- **THEN** logging SHALL indicate the exception was handled
- **AND** recovery actions SHALL be logged
- **AND** severity SHALL reflect actual impact (WARNING if recovered, ERROR if not)

#### Scenario: Unhandled exceptions
- **WHEN** an unhandled exception occurs
- **THEN** the exception SHALL be logged before the application exits
- **AND** full context SHALL be captured for debugging
- **AND** log SHALL be written even if other logging fails

### Requirement: Performance and Efficiency

The application SHALL implement logging with minimal performance overhead.

#### Scenario: Lazy evaluation
- **WHEN** logs are below the configured level
- **THEN** log message processing SHALL be skipped
- **AND** expensive operations in log context SHALL not execute
- **AND** logging SHALL NOT significantly impact application performance

#### Scenario: Structured processor efficiency
- **WHEN** logs are processed
- **THEN** processor pipeline SHALL be efficient
- **AND** processors SHALL avoid redundant operations
- **AND** JSON serialization SHALL be optimized
- **AND** logging overhead SHALL be measurable and minimal (<5% runtime)

#### Scenario: Context binding performance
- **WHEN** context is bound to loggers
- **THEN** binding SHALL be a lightweight operation
- **AND** bound loggers SHALL not duplicate context unnecessarily
- **AND** context SHALL be efficiently passed through call stack

### Requirement: Integration with Monitoring Tools

The application SHALL produce logs compatible with common monitoring and observability platforms.

#### Scenario: JSON log format compatibility
- **WHEN** JSON logging is enabled
- **THEN** logs SHALL be compatible with Elasticsearch/ELK stack
- **AND** logs SHALL be compatible with Datadog
- **AND** logs SHALL be compatible with AWS CloudWatch
- **AND** logs SHALL be compatible with Splunk
- **AND** standard fields (timestamp, level, message) SHALL be present

#### Scenario: Structured data for metrics
- **WHEN** logs contain numeric data
- **THEN** metrics SHALL be extractable from log fields
- **AND** aggregations SHALL be possible (avg BPM, processing time, etc.)
- **AND** time-series data SHALL be queryable
- **AND** dashboards SHALL be buildable from log data

#### Scenario: Log correlation
- **WHEN** analyzing a single operation across multiple logs
- **THEN** logs SHALL be correlatable via context fields
- **AND** operation IDs SHALL persist through related logs
- **AND** file operations SHALL be traceable by filepath
- **AND** request traces SHALL be followable through system

### Requirement: Development Experience

The application SHALL provide excellent logging development experience with clear, readable output and good debugging information.

#### Scenario: Readable development logs
- **WHEN** developers run the application locally
- **THEN** log output SHALL be easy to read and scan
- **AND** important information SHALL be highlighted
- **AND** log levels SHALL be color-coded
- **AND** context SHALL be clearly separated from messages
- **AND** excessive verbosity SHALL be avoided at INFO level

#### Scenario: Debugging with DEBUG level
- **WHEN** DEBUG log level is enabled
- **THEN** detailed diagnostic information SHALL be logged
- **AND** internal state SHALL be visible
- **AND** operation flow SHALL be traceable
- **AND** performance characteristics SHALL be measurable
- **AND** DEBUG logs SHALL NOT leak sensitive information

#### Scenario: Log documentation
- **WHEN** developers need to add logging
- **THEN** documentation SHALL provide clear examples
- **AND** conventions SHALL be explained
- **AND** common patterns SHALL be documented
- **AND** integration with IDE SHALL work well (autocomplete, etc.)

### Requirement: Security and Privacy

The application SHALL ensure logs do not contain sensitive information and are safely handled.

#### Scenario: Sensitive data exclusion
- **WHEN** logging operations involving credentials or keys
- **THEN** sensitive values SHALL NOT be logged
- **AND** API keys SHALL be redacted or omitted
- **AND** passwords SHALL never appear in logs
- **AND** tokens SHALL be truncated or masked if logged
- **AND** PII (personally identifiable information) SHALL be avoided

#### Scenario: Log file security
- **WHEN** writing logs to files
- **THEN** log files SHALL have appropriate permissions (user-only read/write)
- **AND** log files SHALL be stored in secure locations
- **AND** log files SHALL not be world-readable
- **AND** log rotation SHALL securely handle old logs

#### Scenario: Error message sanitization
- **WHEN** logging error messages from external sources
- **THEN** error messages SHALL be sanitized if needed
- **AND** stack traces SHALL not leak security-sensitive paths
- **AND** database errors SHALL not expose credentials
- **AND** logs SHALL be safe to share for debugging

### Requirement: Migration from stdlib logging

The application SHALL completely migrate from Python's standard logging to structlog consistently across the codebase.

#### Scenario: Complete logger replacement
- **WHEN** the migration is complete
- **THEN** all `logging.getLogger()` calls SHALL be replaced with `structlog.get_logger()`
- **AND** no direct `logging` module usage SHALL remain
- **AND** all log calls SHALL use structured format
- **AND** string formatting in log messages SHALL be eliminated

#### Scenario: Backwards compatibility during migration
- **WHEN** migration is in progress
- **THEN** structlog SHALL intercept stdlib logging if needed
- **AND** old and new logging SHALL coexist temporarily
- **AND** consistent format SHALL be maintained across all logs
- **AND** migration SHALL not break existing functionality

#### Scenario: Configuration centralization
- **WHEN** logging is configured
- **THEN** configuration SHALL be in a single module (`src/edm/logging.py`)
- **AND** configuration SHALL be called once at application startup
- **AND** all modules SHALL use the configured logging
- **AND** no module SHALL reconfigure logging independently
