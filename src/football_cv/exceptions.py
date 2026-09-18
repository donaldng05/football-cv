"""
Domain-specific exceptions for football_cv.
"""


class FootballCVError(Exception):
    """Base exception for all football_cv errors."""


class ConfigurationError(FootballCVError):
    """Raised when configuration validation or file parsing fails."""


class ValidationError(FootballCVError):
    """Raised when pre-flight asset/system validation fails."""


class ModelError(FootballCVError):
    """Raised when model loading, weight integrity, or inference fails."""


class VideoProcessingError(FootballCVError):
    """Raised when reading or writing video streams encounters an error."""


class TrackingError(FootballCVError):
    """Raised when multi-object tracking encounters an invalid state."""


class AnalyticsError(FootballCVError):
    """Raised when calculating higher-order football analytics fails."""


class BenchmarkError(FootballCVError):
    """Raised when running or exporting pipeline benchmarks fails."""
