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
        out_of_bounds_policy: str = "strict",
    ):
        if native_core is None:
            raise RuntimeError("Native C++ vision core is not available")
        self.core = native_core.PerspectiveTransformer(
            pixel_vertices, court_width, court_length
        )
        self.court_width = court_width
        self.court_length = court_length
        self.out_of_bounds_policy = out_of_bounds_policy.lower()
        self.pixel_vertices = np.array(
            [[pt.x, pt.y] for pt in self.core.pixel_vertices], dtype=np.float32
        )
        self.perspective_transformer = self.core.homography_matrix

    def get_effective_homography(
        self, camera_matrix: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute effective projective homography H_{t -> pitch} = H_{0 -> pitch} @ H_{t -> 0}.
        """
        if camera_matrix is None:
            return self.perspective_transformer
        return (self.perspective_transformer @ camera_matrix.astype(np.float32)).astype(
            np.float32
        )

    def transform_point(
        self,
        point: Sequence[float] | np.ndarray,
        out_of_bounds_policy: str | None = None,
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Transform a 2D point from broadcast pixel coordinates to metric pitch coordinates.

        Returns None if the point lies outside the calibrated pitch boundary polygon in strict mode.
        """
        policy = (out_of_bounds_policy or self.out_of_bounds_policy).lower()
        if camera_matrix is not None:
            import cv2

            pt_x, pt_y = float(point[0]), float(point[1])
            h_eff = self.get_effective_homography(camera_matrix)
            w_prime = float(h_eff[2, 0] * pt_x + h_eff[2, 1] * pt_y + h_eff[2, 2])
            if w_prime <= 1e-6:
                return None
            reshaped = np.array([[pt_x, pt_y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(reshaped, h_eff)
            res = transformed.reshape(-1, 2)
            is_inside = (
                0.0 <= res[0, 0] <= self.court_width
                and 0.0 <= res[0, 1] <= self.court_length
            )
            if not is_inside and policy == "strict":
                return None
            if not is_inside and policy == "clip":
                res[0, 0] = float(np.clip(res[0, 0], 0.0, self.court_width))
                res[0, 1] = float(np.clip(res[0, 1], 0.0, self.court_length))
            return res

        if policy == "strict":
            pt = self.core.transform_point(point, True)
            if pt is None:
                return None
            return np.array([[pt.x, pt.y]], dtype=np.float32)

        # In non-strict mode (clip or extrapolate):
        pt = self.core.transform_point(point, False)
        if pt is None:
            return None
        tx, ty = pt.x, pt.y
        if policy == "clip":
            tx = float(np.clip(tx, 0.0, self.court_width))
            ty = float(np.clip(ty, 0.0, self.court_length))
        return np.array([[tx, ty]], dtype=np.float32)

    def add_transformed_position_to_tracks(
        self,
        tracks: dict[str, Any],
        camera_matrices: list[np.ndarray] | None = None,
    ) -> None:
        """Apply perspective transformation to object positions in tracks."""
        use_matrices = camera_matrices is not None and len(camera_matrices) > 0

        for obj_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                cam_mat = (
                    camera_matrices[frame_num]
                    if use_matrices and frame_num < len(camera_matrices)
                    else None
                )
                for track_id, track_info in track.items():
                    tracks[obj_name][frame_num][track_id]["position_transformed"] = None

                    if cam_mat is not None and "position" in track_info:
                        pos = track_info["position"]
                        transformed = self.transform_point(pos, camera_matrix=cam_mat)
                    elif "position_adjusted" in track_info:
                        pos = track_info["position_adjusted"]
                        transformed = self.transform_point(pos)
                    elif "position" in track_info:
                        pos = track_info["position"]
                        transformed = self.transform_point(pos)
                    else:
                        continue

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
        margin_ratio_x: float = 0.05,
        margin_ratio_y: float = 0.10,
        use_dynamic_margins: bool = True,
    ):
        if native_core is None:
            raise RuntimeError("Native C++ vision core is not available")
        import cv2

        self.core = native_core.CameraMotionEstimator(
            minimum_distance, scene_cut_threshold
        )
        self.minimum_distance = minimum_distance
        self.scene_cut_threshold = scene_cut_threshold
        self.margin_ratio_x = margin_ratio_x
        self.margin_ratio_y = margin_ratio_y
        self.use_dynamic_margins = use_dynamic_margins

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        h, w = first_gray.shape[:2]
        mask_features = np.zeros_like(first_gray)
        if use_dynamic_margins and w > 100:
            left_w = max(20, int(w * margin_ratio_x))
            right_w = max(20, int(w * margin_ratio_x))
            top_h = max(20, int(h * margin_ratio_y))
            mask_features[:, :left_w] = 1
            mask_features[:, w - right_w :] = 1
            mask_features[:top_h, :] = 1
        else:
            mask_features[:, 0:20] = 1
            mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )
        self._last_gray: np.ndarray | None = None
        self._last_features: np.ndarray | None = None
        self._cumulative_movement: tuple[float, float] = (0.0, 0.0)
        self._cumulative_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self.camera_matrices: list[np.ndarray] = [np.eye(3, dtype=np.float32)]
        self.last_chunk_matrices: list[np.ndarray] = []

    def reset(self) -> None:
        """Reset internal streaming camera state."""
        self._last_gray = None
        self._last_features = None
        self._cumulative_movement = (0.0, 0.0)
        self._cumulative_matrix = np.eye(3, dtype=np.float32)
        self.camera_matrices = [np.eye(3, dtype=np.float32)]
        self.last_chunk_matrices = []

    def estimate_chunk(self, frames: list[np.ndarray]) -> list[tuple[float, float]]:
        """
        Estimate frame-by-frame camera translation (dx, dy) across a chunk of frames.
        Delegates displacement calculations and scene cut detection to native C++ core.
        """
        import cv2

        displacements: list[tuple[float, float]] = []
        chunk_matrices: list[np.ndarray] = []
        if not frames:
            self.last_chunk_matrices = []
            return displacements

        for frame in frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self._last_gray is None:
                self._last_gray = frame_gray
                self._last_features = cv2.goodFeaturesToTrack(
                    self._last_gray, **self.features
                )
                displacements.append((0.0, 0.0))
                chunk_matrices.append(self._cumulative_matrix.copy())
                continue

            if self._last_features is None or len(self._last_features) == 0:
                self._last_features = cv2.goodFeaturesToTrack(
                    self._last_gray, **self.features
                )
                if self._last_features is None or len(self._last_features) == 0:
                    displacements.append((0.0, 0.0))
                    chunk_matrices.append(self._cumulative_matrix.copy())
                    self._last_gray = frame_gray
                    continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                self._last_gray, frame_gray, self._last_features, None, **self.lk_params
            )

            cam_dx, cam_dy = 0.0, 0.0
            is_cut = False
            h_step = np.eye(3, dtype=np.float32)

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = self._last_features[status == 1]
                motion = self.core.estimate_from_features(good_old, good_new)
                cam_dx, cam_dy = motion.dx, motion.dy
                is_cut = motion.is_scene_cut

                if not is_cut and len(good_new) >= 4 and len(good_old) >= 4:
                    affine_mat, inliers = cv2.estimateAffinePartial2D(
                        good_new, good_old, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if affine_mat is not None and (inliers is None or np.sum(inliers) >= 3):
                        h_step[:2, :] = affine_mat.astype(np.float32)
                    else:
                        h_step[0, 2] = float(cam_dx)
                        h_step[1, 2] = float(cam_dy)
                elif not is_cut:
                    h_step[0, 2] = float(cam_dx)
                    h_step[1, 2] = float(cam_dy)

            if is_cut:
                displacements.append((0.0, 0.0))
                self._cumulative_movement = (0.0, 0.0)
                self._cumulative_matrix = np.eye(3, dtype=np.float32)
                chunk_matrices.append(self._cumulative_matrix.copy())
                self._last_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features
                )
            else:
                displacements.append((cam_dx, cam_dy))
                self._cumulative_movement = (
                    self._cumulative_movement[0] + cam_dx,
                    self._cumulative_movement[1] + cam_dy,
                )
                self._cumulative_matrix = self._cumulative_matrix @ h_step
                chunk_matrices.append(self._cumulative_matrix.copy())
                if cam_dx != 0.0 or cam_dy != 0.0:
                    self._last_features = cv2.goodFeaturesToTrack(
                        frame_gray, **self.features
                    )

            self._last_gray = frame_gray

        self.last_chunk_matrices = chunk_matrices
        return displacements

    def add_adjust_positions_to_chunk(
        self,
        tracks: dict[str, Any],
        camera_movement_chunk: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Adjust object positions in chunk tracks using cumulative motion.
        Returns the cumulative motion for each frame in the chunk.
        """
        total_dx = sum(m[0] for m in camera_movement_chunk)
        total_dy = sum(m[1] for m in camera_movement_chunk)
        base_x = self._cumulative_movement[0] - total_dx
        base_y = self._cumulative_movement[1] - total_dy

        cum_list: list[tuple[float, float]] = []
        for m in camera_movement_chunk:
            base_x += m[0]
            base_y += m[1]
            cum_list.append((base_x, base_y))

        for _obj_name, object_tracks in tracks.items():
            for frame_idx, track in enumerate(object_tracks):
                if frame_idx >= len(cum_list):
                    continue
                cam_cum = cum_list[frame_idx]
                for _track_id, track_info in track.items():
                    if "position" not in track_info:
                        continue
                    pos = track_info["position"]
                    track_info["position_adjusted"] = (
                        float(pos[0] - cam_cum[0]),
                        float(pos[1] - cam_cum[1]),
                    )
        return cum_list

    @staticmethod
    def accumulate_motion(
        motion_steps: Sequence[Any],
    ) -> list[tuple[float, float]]:
        """Accumulate frame-to-frame camera displacements into cumulative translation relative to frame 0."""
        cumulative: list[tuple[float, float]] = []
        total_dx = 0.0
        total_dy = 0.0
        for step in motion_steps:
            if hasattr(step, "dx") and hasattr(step, "dy"):
                dx, dy = float(step.dx), float(step.dy)
                is_cut = getattr(step, "is_scene_cut", False)
            elif isinstance(step, (list, tuple)):
                dx, dy = float(step[0]), float(step[1])
                is_cut = False
            else:
                dx, dy = 0.0, 0.0
                is_cut = False

            if is_cut:
                total_dx = 0.0
                total_dy = 0.0
            else:
                total_dx += dx
                total_dy += dy
            cumulative.append((total_dx, total_dy))
        return cumulative

    def add_adjust_positions_to_tracks(
        self,
        tracks: dict[str, Any],
        camera_movement_per_frame: list[list[float] | tuple[float, float]],
        cumulative: bool = False,
    ) -> None:
        """Adjust object positions to compensate for camera motion."""
        movements = (
            self.accumulate_motion(camera_movement_per_frame)
            if cumulative
            else camera_movement_per_frame
        )
        for obj_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if frame_num >= len(movements):
                    continue
                cam_movement = movements[frame_num]
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
                movement = pickle.load(f)
            matrices = [np.eye(3, dtype=np.float32)]
            running_m = np.eye(3, dtype=np.float32)
            for item in movement[1:]:
                dx, dy = float(item[0]), float(item[1])
                step_m = np.eye(3, dtype=np.float32)
                step_m[0, 2] = dx
                step_m[1, 2] = dy
                running_m = running_m @ step_m
                matrices.append(running_m.copy())
            self.camera_matrices = matrices
            return movement

        camera_movement: list[tuple[float, float]] = [(0.0, 0.0)]
        camera_matrices: list[np.ndarray] = [np.eye(3, dtype=np.float32)]
        cum_matrix = np.eye(3, dtype=np.float32)
        if not frames:
            self.camera_matrices = camera_matrices
            return camera_movement

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                if old_features is None or len(old_features) == 0:
                    camera_movement.append((0.0, 0.0))
                    camera_matrices.append(cum_matrix.copy())
                    old_gray = frame_gray
                    continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            cam_dx, cam_dy = 0.0, 0.0
            is_cut = False
            h_step = np.eye(3, dtype=np.float32)

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                # Delegate feature displacement & scene cut analysis to compiled C++ core
                motion = self.core.estimate_from_features(good_old, good_new)
                cam_dx, cam_dy = motion.dx, motion.dy
                is_cut = motion.is_scene_cut

                if not is_cut and len(good_new) >= 4 and len(good_old) >= 4:
                    affine_mat, inliers = cv2.estimateAffinePartial2D(
                        good_new, good_old, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if affine_mat is not None and (inliers is None or np.sum(inliers) >= 3):
                        h_step[:2, :] = affine_mat.astype(np.float32)
                    else:
                        h_step[0, 2] = float(cam_dx)
                        h_step[1, 2] = float(cam_dy)
                elif not is_cut:
                    h_step[0, 2] = float(cam_dx)
                    h_step[1, 2] = float(cam_dy)

            if is_cut:
                camera_movement.append((0.0, 0.0))
                cum_matrix = np.eye(3, dtype=np.float32)
                camera_matrices.append(cum_matrix.copy())
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            else:
                camera_movement.append((cam_dx, cam_dy))
                cum_matrix = cum_matrix @ h_step
                camera_matrices.append(cum_matrix.copy())
                if cam_dx != 0.0 or cam_dy != 0.0:
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray

        self.camera_matrices = camera_matrices

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
    out_of_bounds_policy: str = "strict",
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
        out_of_bounds_policy: 'strict', 'clip', or 'extrapolate'.
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
            out_of_bounds_policy=out_of_bounds_policy,
        )

    from ..perspective.transformer import PerspectiveTransformer

    return PerspectiveTransformer(
        pixel_vertices=pixel_vertices,
        court_width=court_width,
        court_length=court_length,
        out_of_bounds_policy=out_of_bounds_policy,
    )


def get_camera_motion_estimator(
    first_frame: np.ndarray,
    minimum_distance: float = 5.0,
    scene_cut_threshold: float = 80.0,
    margin_ratio_x: float = 0.05,
    margin_ratio_y: float = 0.10,
    use_dynamic_margins: bool = True,
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
        margin_ratio_x: Horizontal margin ratio for feature tracking.
        margin_ratio_y: Vertical margin ratio for feature tracking.
        use_dynamic_margins: Whether to scale margins dynamically with resolution.
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
            margin_ratio_x=margin_ratio_x,
            margin_ratio_y=margin_ratio_y,
            use_dynamic_margins=use_dynamic_margins,
        )

    from ..camera_motion.estimator import CameraMotionEstimator

    return CameraMotionEstimator(
        first_frame=first_frame,
        minimum_distance=minimum_distance,
        scene_cut_threshold=scene_cut_threshold,
        margin_ratio_x=margin_ratio_x,
        margin_ratio_y=margin_ratio_y,
        use_dynamic_margins=use_dynamic_margins,
    )
