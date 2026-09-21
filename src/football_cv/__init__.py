"""
football_cv: Computer Vision & Advanced Football Analytics Pipeline.
"""

import os
import sys

# On Windows, add package directory to DLL search path so _core can load bundled DLLs (e.g. onnxruntime.dll)
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _pkg_dir = os.path.dirname(__file__)
    if os.path.isdir(_pkg_dir):
        try:
            os.add_dll_directory(_pkg_dir)
        except OSError:
            pass

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
