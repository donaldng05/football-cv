"""
Camera motion estimation using sparse optical flow.
"""

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.geometry import measure_distance


class CameraMotionEstimator:
    """Estimates pan and tilt motion using Lucas-Kanade optical flow on perimeter pixels."""

    def __init__(
        self,
        first_frame: np.ndarray,
        minimum_distance: float = 5.0,
        scene_cut_threshold: float = 80.0,
        margin_ratio_x: float = 0.05,
        margin_ratio_y: float = 0.10,
        use_dynamic_margins: bool = True,
    ):
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
            # Fallback legacy margins
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

    @staticmethod
    def create_background_mask(
        frame_shape: tuple[int, int] | tuple[int, int, int],
        entity_bboxes: Sequence[Sequence[float]] | None = None,
        margin: int = 15,
    ) -> np.ndarray:
        """
        Create a binary mask for optical flow feature tracking excluding moving players and objects.
        """
        h, w = frame_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)
        if entity_bboxes:
            for bbox in entity_bboxes:
                if len(bbox) >= 4:
                    x1 = max(0, int(bbox[0]) - margin)
                    y1 = max(0, int(bbox[1]) - margin)
                    x2 = min(w, int(bbox[2]) + margin)
                    y2 = min(h, int(bbox[3]) + margin)
                    mask[y1:y2, x1:x2] = 0
        return mask

    @staticmethod
    def _estimate_inter_frame_matrix(
        good_new: np.ndarray,
        good_old: np.ndarray,
        cam_dx: float,
        cam_dy: float,
        is_cut: bool,
    ) -> np.ndarray:
        """
        Estimate 3x3 projective transformation H_{t -> t-1} mapping points from
        current frame t into previous frame t-1 coordinate space.
        """
        if is_cut:
            return np.eye(3, dtype=np.float32)

        if len(good_new) >= 4 and len(good_old) >= 4:
            affine_mat, inliers = cv2.estimateAffinePartial2D(
                good_new, good_old, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if affine_mat is not None and (inliers is None or np.sum(inliers) >= 3):
                h = np.eye(3, dtype=np.float32)
                h[:2, :] = affine_mat.astype(np.float32)
                return h

        h = np.eye(3, dtype=np.float32)
        h[0, 2] = float(cam_dx)
        h[1, 2] = float(cam_dy)
        return h

    def estimate_chunk(self, frames: list[np.ndarray]) -> list[tuple[float, float]]:
        """
        Estimate frame-by-frame camera translation (dx, dy) across a chunk of frames.
        Maintains internal state across chunks with O(1) memory.
        """
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

                if len(good_new) > 0:
                    dxs = [
                        float(old[0] - new[0])
                        for old, new in zip(good_old, good_new, strict=True)
                    ]
                    dys = [
                        float(old[1] - new[1])
                        for old, new in zip(good_old, good_new, strict=True)
                    ]
                    dists = [
                        float(measure_distance(old.ravel(), new.ravel()))
                        for old, new in zip(good_old, good_new, strict=True)
                    ]

                    med_dist = float(np.median(dists))
                    if med_dist > self.scene_cut_threshold:
                        is_cut = True
                    else:
                        dx = float(np.median(dxs))
                        dy = float(np.median(dys))
                        disp_mag = float(np.sqrt(dx**2 + dy**2))
                        if (
                            disp_mag > self.minimum_distance
                            or med_dist > self.minimum_distance
                        ):
                            cam_dx, cam_dy = dx, dy

                    h_step = self._estimate_inter_frame_matrix(
                        good_new, good_old, cam_dx, cam_dy, is_cut
                    )

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
    def filter_margin_points(
        points: Sequence[Sequence[float] | tuple[float, float]],
        frame_width: float,
        left_margin_width: float = 20.0,
        right_margin_start: float = 900.0,
        right_margin_end: float = 1050.0,
    ) -> list[tuple[float, float]]:
        """Filter points to isolate vertical broadcast margins (excluding players on pitch)."""
        filtered = []
        for pt in points:
            px, py = float(pt[0]), float(pt[1])
            in_left = 0.0 <= px <= left_margin_width
            in_right = right_margin_start <= px <= right_margin_end
            if in_left or in_right:
                filtered.append((px, py))
        return filtered

    @staticmethod
    def accumulate_motion(
        motion_steps: Sequence[Any],
    ) -> list[tuple[float, float]]:
        """
        Accumulate frame-to-frame camera displacements into cumulative translation relative to frame 0.
        """
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

        Args:
            frames: List of BGR frames.
            read_from_stub: If True and stub exists, load from pickle.
            stub_path: Filepath for stub cache.

        Returns:
            List of (dx, dy) displacements for each frame.
        """
        if read_from_stub and stub_path is not None and Path(stub_path).is_file():
            with open(stub_path, "rb") as f:
                movement = pickle.load(f)
            # Reconstruct camera_matrices from displacement stubs
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

                if len(good_new) > 0:
                    dxs = [
                        float(old[0] - new[0])
                        for old, new in zip(good_old, good_new, strict=True)
                    ]
                    dys = [
                        float(old[1] - new[1])
                        for old, new in zip(good_old, good_new, strict=True)
                    ]
                    dists = [
                        float(measure_distance(old.ravel(), new.ravel()))
                        for old, new in zip(good_old, good_new, strict=True)
                    ]

                    med_dist = float(np.median(dists))
                    if med_dist > self.scene_cut_threshold:
                        # Sudden flow jump indicates a broadcast scene cut
                        is_cut = True
                    else:
                        dx = float(np.median(dxs))
                        dy = float(np.median(dys))
                        disp_mag = float(np.sqrt(dx**2 + dy**2))
                        if (
                            disp_mag > self.minimum_distance
                            or med_dist > self.minimum_distance
                        ):
                            cam_dx, cam_dy = dx, dy
                        elif med_dist > self.minimum_distance:
                            best_idx = int(
                                np.argmin([abs(d - med_dist) for d in dists])
                            )
                            cam_dx, cam_dy = dxs[best_idx], dys[best_idx]

                    h_step = self._estimate_inter_frame_matrix(
                        good_new, good_old, cam_dx, cam_dy, is_cut
                    )

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


# Alias for backward compatibility
CameraMovementEstimator = CameraMotionEstimator
