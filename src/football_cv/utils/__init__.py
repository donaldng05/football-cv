"""
Utility modules for validation, geometry, and video I/O.
"""

from .validation import (
    run_preflight_checks,
    validate_device,
    validate_model_path,
    validate_video_path,
)

__all__ = [
    "run_preflight_checks",
    "validate_device",
    "validate_model_path",
    "validate_video_path",
]
