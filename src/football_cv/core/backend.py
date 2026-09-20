"""
Vision backend abstraction and execution runtime resolver.

Allows the football-cv pipeline to switch seamlessly between pure-Python
and native C++ vision backends (football_cv._core) with fallback handling.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

BackendType = Literal["python", "cpp"]

# Attempt to import compiled native C++ vision core
try:
    from .. import _core as native_core  # type: ignore

    _HAS_CPP_CORE = True
except ImportError:
    native_core = None
    _HAS_CPP_CORE = False


def has_cpp_core() -> bool:
    """Return True if compiled C++ vision core (_core) is available."""
    return _HAS_CPP_CORE


def resolve_backend(
    requested: str | None = None,
    *,
    strict: bool = False,
) -> BackendType:
    """
    Resolve requested backend against available runtime backends.

    Args:
        requested: "python" or "cpp" (case-insensitive). Defaults to "python".
        strict: If True, raises ConfigurationError if "cpp" is requested but unavailable.
                If False, logs a warning and falls back to "python".

    Returns:
        Resolved backend ("python" or "cpp").

    Raises:
        ConfigurationError: If an invalid backend name is specified or if strict=True
                            and the C++ backend is not compiled.
    """
    if requested is None:
        return "python"

    req_normalized = requested.strip().lower()
    valid_backends = {"python", "cpp"}
    if req_normalized not in valid_backends:
        raise ConfigurationError(
            f"Unsupported vision backend '{requested}'. Must be one of {valid_backends}"
        )

    if req_normalized == "cpp":
        if has_cpp_core():
            return "cpp"
        if strict:
            raise ConfigurationError(
                "C++ vision backend requested, but native module 'football_cv._core' "
                "is not compiled or installed."
            )
        logger.warning(
            "C++ vision backend requested, but 'football_cv._core' is not available. "
            "Falling back to pure-Python backend."
        )
        return "python"

    return "python"


def get_perspective_transformer(
    pixel_vertices: Sequence[Sequence[float]] | None = None,
    court_width: float = 68.0,
    court_length: float = 23.32,
    backend: str = "python",
    *,
    strict: bool = False,
) -> Any:
    """
    Instantiate a PerspectiveTransformer using the resolved backend.

    Args:
        pixel_vertices: 4-point broadcast polygon coordinates.
        court_width: Target pitch width in meters.
        court_length: Target pitch length in meters.
        backend: "python" or "cpp".
        strict: Whether to error if requested backend is unavailable.

    Returns:
        PerspectiveTransformer instance compatible with football-cv pipeline.
    """
    resolved = resolve_backend(backend, strict=strict)

    if resolved == "cpp" and native_core is not None:
        # Native C++ transformer wrapper will be wired here in Phase 3
        return native_core.PerspectiveTransformer(
            pixel_vertices, court_width, court_length
        )

    from ..perspective.transformer import PerspectiveTransformer

    return PerspectiveTransformer(
        pixel_vertices=pixel_vertices,
        court_width=court_width,
        court_length=court_length,
    )


def get_camera_motion_estimator(
    first_frame: np.ndarray,
    minimum_distance: float = 5.0,
    scene_cut_threshold: float = 80.0,
    backend: str = "python",
    *,
    strict: bool = False,
) -> Any:
    """
    Instantiate a CameraMotionEstimator using the resolved backend.

    Args:
        first_frame: Initial video frame.
        minimum_distance: Minimum feature displacement threshold in pixels.
        scene_cut_threshold: Maximum displacement before declaring a scene cut.
        backend: "python" or "cpp".
        strict: Whether to error if requested backend is unavailable.

    Returns:
        CameraMotionEstimator instance compatible with football-cv pipeline.
    """
    resolved = resolve_backend(backend, strict=strict)

    if resolved == "cpp" and native_core is not None:
        # Native C++ estimator wrapper will be wired here in Phase 3
        return native_core.CameraMotionEstimator(
            first_frame, minimum_distance, scene_cut_threshold
        )

    from ..camera_motion.estimator import CameraMotionEstimator

    return CameraMotionEstimator(
        first_frame=first_frame,
        minimum_distance=minimum_distance,
        scene_cut_threshold=scene_cut_threshold,
    )
