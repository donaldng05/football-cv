"""
football_cv: Computer Vision & Advanced Football Analytics Pipeline.
"""

__version__ = "0.1.0"
__author__ = "Quy Duong"

from .config import AppConfig, VisionConfig, load_config
from .exceptions import (
    ConfigurationError,
    FootballCVError,
    ModelError,
    ValidationError,
    VideoProcessingError,
)
from .logging_config import setup_logging

__all__ = [
    "AppConfig",
    "ConfigurationError",
    "FootballCVError",
    "ModelError",
    "ValidationError",
    "VideoProcessingError",
    "VisionConfig",
    "load_config",
    "setup_logging",
]
