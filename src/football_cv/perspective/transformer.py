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
    ):
        self.court_width = court_width
        self.court_length = court_length

        if pixel_vertices is None:
            pixel_vertices = [
                [110.0, 1035.0],
                [265.0, 275.0],
                [910.0, 260.0],
                [1640.0, 915.0],
            ]

        self.pixel_vertices = np.array(pixel_vertices, dtype=np.float32)

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

    def transform_point(self, point: Sequence[float] | np.ndarray) -> np.ndarray | None:
        """
        Transform a 2D point from broadcast pixel coordinates to metric pitch coordinates.

        Returns None if the point lies outside the calibrated pitch boundary polygon.
        """
        pt = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, pt, False) >= 0
        if not is_inside:
            return None

        reshaped = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.perspective_transformer)
        return transformed.reshape(-1, 2)

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


# Alias for backward compatibility
ViewTransformer = PerspectiveTransformer
