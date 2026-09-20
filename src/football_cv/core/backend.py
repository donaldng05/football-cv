"""
Vision backend abstraction and execution runtime resolver.

Allows the football-cv pipeline to switch seamlessly between pure-Python
and native C++ vision backends (football_cv._core) with fallback handling.
"""

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
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


class CppPerspectiveTransformerAdapter:
    """Adapter wrapping native C++ PerspectiveTransformer for pipeline compatibility."""

    def __init__(
        self,
        pixel_vertices: Sequence[Sequence[float]] | None = None,
        court_width: float = 68.0,
        court_length: float = 23.32,
    ):
        if native_core is None:
            raise RuntimeError("Native C++ vision core is not available")
        self.core = native_core.PerspectiveTransformer(
            pixel_vertices, court_width, court_length
        )
        self.court_width = court_width
        self.court_length = court_length
        self.pixel_vertices = np.array(
            [[pt.x, pt.y] for pt in self.core.pixel_vertices], dtype=np.float32
        )
        self.perspective_transformer = self.core.homography_matrix

    def transform_point(self, point: Sequence[float] | np.ndarray) -> np.ndarray | None:
        """
        Transform a 2D point from broadcast pixel coordinates to metric pitch coordinates.

        Returns None if the point lies outside the calibrated pitch boundary polygon.
        """
        pt = self.core.transform_point(point)
        if pt is None:
            return None
        return np.array([[pt.x, pt.y]], dtype=np.float32)

    def add_transformed_position_to_tracks(self, tracks: dict[str, Any]) -> None:
        """Apply perspective transformation to object positions in tracks."""
        for obj_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    tracks[obj_name][frame_num][track_id]["position_transformed"] = None

                    if "position_adjusted" not in track_info:
                        continue

                    pos = track_info["position_adjusted"]
                    transformed = self.transform_point(pos)

                    if transformed is not None:
                        tracks[obj_name][frame_num][track_id][
                            "position_transformed"
                        ] = transformed.squeeze().tolist()


class CppCameraMotionEstimatorAdapter:
    """Adapter wrapping native C++ CameraMotionEstimator for pipeline compatibility."""

    def __init__(
        self,
        first_frame: np.ndarray,
        minimum_distance: float = 5.0,
        scene_cut_threshold: float = 80.0,
    ):
        if native_core is None:
            raise RuntimeError("Native C++ vision core is not available")
        import cv2

        self.core = native_core.CameraMotionEstimator(
            minimum_distance, scene_cut_threshold
        )
        self.minimum_distance = minimum_distance
        self.scene_cut_threshold = scene_cut_threshold

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_gray)
        # Mask features: isolate vertical borders to exclude players running on the pitch
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(
        self,
        tracks: dict[str, Any],
        camera_movement_per_frame: list[list[float] | tuple[float, float]],
    ) -> None:
        """Adjust object positions to compensate for camera motion."""
        for obj_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                cam_movement = camera_movement_per_frame[frame_num]
                for track_id, track_info in track.items():
                    if "position" not in track_info:
                        continue
                    pos = track_info["position"]
                    pos_adjusted = (
                        float(pos[0] - cam_movement[0]),
                        float(pos[1] - cam_movement[1]),
                    )
                    tracks[obj_name][frame_num][track_id]["position_adjusted"] = (
                        pos_adjusted
                    )

    def get_camera_movement(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> list[tuple[float, float]]:
        """
        Estimate frame-by-frame camera translation (dx, dy).

        Delegates displacement calculations and scene cut detection to native C++ core.
        """
        import cv2

        if read_from_stub and stub_path is not None and Path(stub_path).is_file():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement: list[tuple[float, float]] = [(0.0, 0.0)]
        if not frames:
            return camera_movement

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                if old_features is None or len(old_features) == 0:
                    camera_movement.append((0.0, 0.0))
                    old_gray = frame_gray
                    continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            cam_dx, cam_dy = 0.0, 0.0
            is_cut = False

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                # Delegate feature displacement & scene cut analysis to compiled C++ core
                motion = self.core.estimate_from_features(good_old, good_new)
                cam_dx, cam_dy = motion.dx, motion.dy
                is_cut = motion.is_scene_cut

            if is_cut:
                camera_movement.append((0.0, 0.0))
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            elif cam_dx != 0.0 or cam_dy != 0.0:
                camera_movement.append((cam_dx, cam_dy))
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            else:
                camera_movement.append((0.0, 0.0))

            old_gray = frame_gray

        if stub_path is not None:
            stub_p = Path(stub_path)
            stub_p.parent.mkdir(parents=True, exist_ok=True)
            with open(stub_p, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(
        self,
        frames: list[np.ndarray],
        camera_movement_per_frame: list[list[float] | tuple[float, float]],
    ) -> list[np.ndarray]:
        """Backward-compatible wrapper delegating to FrameAnnotator."""
        from ..rendering.annotations import FrameAnnotator

        return FrameAnnotator.draw_camera_movement(frames, camera_movement_per_frame)


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
        return CppPerspectiveTransformerAdapter(
            pixel_vertices=pixel_vertices,
            court_width=court_width,
            court_length=court_length,
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
        return CppCameraMotionEstimatorAdapter(
            first_frame=first_frame,
            minimum_distance=minimum_distance,
            scene_cut_threshold=scene_cut_threshold,
        )

    from ..camera_motion.estimator import CameraMotionEstimator

    return CameraMotionEstimator(
        first_frame=first_frame,
        minimum_distance=minimum_distance,
        scene_cut_threshold=scene_cut_threshold,
    )
