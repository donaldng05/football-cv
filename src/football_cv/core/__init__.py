"""
Core vision engine backend interfaces and native runtime bindings.
"""

from .backend import (
    BackendType,
    get_camera_motion_estimator,
    get_perspective_transformer,
    has_cpp_core,
    resolve_backend,
)

__all__ = [
    "BackendType",
    "get_camera_motion_estimator",
    "get_perspective_transformer",
    "has_cpp_core",
    "resolve_backend",
]
