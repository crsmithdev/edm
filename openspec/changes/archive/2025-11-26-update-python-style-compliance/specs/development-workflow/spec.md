## ADDED Requirements

### Requirement: Python Style Compliance Enforcement
The project SHALL maintain code that conforms to the Python style guide documented in `docs/python-style.md`.

#### Scenario: Modern type hint syntax
- **WHEN** writing type annotations in Python code
- **THEN** code uses modern Python 3.10+ syntax (`list`, `dict`, `| None`) instead of deprecated `typing` imports (`List`, `Dict`, `Optional`)

#### Scenario: Google-style docstrings
- **WHEN** documenting functions, classes, or modules
- **THEN** docstrings follow Google-style format with `Args:`, `Returns:`, `Raises:`, and `Examples:` sections (not Numpy-style with `Parameters\n----------`)

#### Scenario: Structured logging
- **WHEN** adding log statements
- **THEN** code uses structlog with keyword arguments for context (`logger.info("event", key=value)`) instead of f-string interpolation (`logger.info(f"event {value}")`)

#### Scenario: Consistent logging module
- **WHEN** importing logging functionality
- **THEN** code imports `structlog` (not standard library `logging`) for consistency across the codebase
