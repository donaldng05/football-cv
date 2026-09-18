"""
Camera motion estimation using sparse optical flow.
"""

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.geometry import measure_distance, measure_xy_distance


class CameraMotionEstimator:
    """Estimates pan and tilt motion using Lucas-Kanade optical flow on perimeter pixels."""

    def __init__(
        self,
        first_frame: np.ndarray,
        minimum_distance: float = 5.0,
        scene_cut_threshold: float = 80.0,
    ):
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

        Args:
            frames: List of BGR frames.
            read_from_stub: If True and stub exists, load from pickle.
            stub_path: Filepath for stub cache.

        Returns:
            List of (dx, dy) displacements for each frame.
        """
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

            max_distance = 0.0
            cam_dx, cam_dy = 0.0, 0.0

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                for new, old in zip(good_new, good_old, strict=False):
                    dist = measure_distance(new.ravel(), old.ravel())
                    if dist > max_distance:
                        max_distance = dist
                        cam_dx, cam_dy = measure_xy_distance(old.ravel(), new.ravel())

            if max_distance > self.scene_cut_threshold:
                # Sudden flow magnitude jump indicates a scene cut or broadcast discontinuity
                camera_movement.append((0.0, 0.0))
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            elif max_distance > self.minimum_distance:
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


# Alias for backward compatibility
CameraMovementEstimator = CameraMotionEstimator
