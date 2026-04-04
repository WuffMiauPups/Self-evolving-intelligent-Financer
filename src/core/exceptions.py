class ConfigurationError(RuntimeError):
    """Raised when the application configuration is invalid."""


class RiskViolation(RuntimeError):
    """Raised when an unsafe trade is attempted."""


class ExecutionError(RuntimeError):
    """Raised when order execution fails."""


class DataUnavailableError(RuntimeError):
    """Raised when requested market data is unavailable."""

