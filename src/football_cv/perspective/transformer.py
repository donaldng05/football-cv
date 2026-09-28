"""
Perspective transformation and homography to metric pitch coordinates.
"""

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np


class PerspectiveTransformer:
    """Projects pixel coordinates to metric pitch coordinates (meters) via 4-point homography."""

    def __init__(
        self,
        pixel_vertices: Sequence[Sequence[float]] | None = None,
        court_width: float = 68.0,
        court_length: float = 23.32,
        out_of_bounds_policy: str = "strict",
        target_vertices: Sequence[Sequence[float]] | None = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.court_width = court_width
        self.court_length = court_length
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.out_of_bounds_policy = out_of_bounds_policy.lower()

        if pixel_vertices is None:
            pixel_vertices = [
                [110.0, 1035.0],
                [265.0, 275.0],
                [910.0, 260.0],
                [1640.0, 915.0],
            ]

        self.pixel_vertices = np.array(pixel_vertices, dtype=np.float32)

        if target_vertices is not None:
            self.target_vertices = np.array(target_vertices, dtype=np.float32)
        else:
            self.target_vertices = np.array(
                [
                    [0.0, 0.0],
                    [0.0, court_length],
                    [court_width, court_length],
                    [court_width, 0.0],
                ],
                dtype=np.float32,
            )

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

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

        Args:
            point: Pixel coordinates [x, y].
            out_of_bounds_policy: Optional override ('strict', 'clip', 'extrapolate').
            camera_matrix: Optional 3x3 cumulative camera transformation matrix H_{t -> 0}.

        Returns:
            np.ndarray of shape (1, 2) in metric pitch coordinates, or None if outside polygon in strict mode.
        """
        policy = (out_of_bounds_policy or self.out_of_bounds_policy).lower()
        pt_x, pt_y = float(point[0]), float(point[1])

        h_eff = self.get_effective_homography(camera_matrix)

        # Check projective horizon singularity (w' = h31*x + h32*y + h33)
        w_prime = float(h_eff[2, 0] * pt_x + h_eff[2, 1] * pt_y + h_eff[2, 2])
        if w_prime <= 1e-6:
            return None

        reshaped = np.array([[pt_x, pt_y]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, h_eff)
        res = transformed.reshape(-1, 2)

        if camera_matrix is None:
            is_inside = (
                cv2.pointPolygonTest(
                    self.pixel_vertices, (int(pt_x), int(pt_y)), False
                )
                >= 0
            )
        else:
            # When camera motion is compensated, evaluate against invariant pitch metric bounds
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

    def transform_points_batch(
        self,
        points: np.ndarray,
        camera_matrix: np.ndarray | None = None,
        out_of_bounds_policy: str | None = None,
    ) -> np.ndarray:
        """
        Vectorized transformation of an (N, 2) array of pixel coordinates to metric pitch coordinates.
        """
        if points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        h_eff = self.get_effective_homography(camera_matrix)

        reshaped = pts.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, h_eff).reshape(-1, 2)

        policy = (out_of_bounds_policy or self.out_of_bounds_policy).lower()
        if policy == "clip":
            transformed[:, 0] = np.clip(transformed[:, 0], 0.0, self.court_width)
            transformed[:, 1] = np.clip(transformed[:, 1], 0.0, self.court_length)

        return transformed

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


# Alias for backward compatibility
ViewTransformer = PerspectiveTransformer

