"""Custom exceptions for the EDM library."""


class EDMError(Exception):
    """Base exception for all EDM library errors."""

    pass


class AudioFileError(EDMError):
    """Error loading or reading audio files."""

    pass


class AnalysisError(EDMError):
    """Error during analysis operations."""

    pass


class ExternalServiceError(EDMError):
    """Error communicating with external services."""

    pass


class ModelNotFoundError(EDMError):
    """Requested model not found."""

    pass


class ConfigurationError(EDMError):
    """Configuration validation or loading error."""

    pass
