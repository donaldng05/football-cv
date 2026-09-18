"""
Ball position interpolation for bridging missing detection frames.
"""

from typing import Any

import pandas as pd


class BallInterpolator:
    """Interpolates missing ball detections using linear temporal interpolation."""

    @staticmethod
    def interpolate_ball_positions(
        ball_positions: list[dict[int, dict[str, Any]]],
        limit: int = 10,
    ) -> list[dict[int, dict[str, Any]]]:
        """
        Interpolate missing ball coordinates across consecutive frames.

        Args:
            ball_positions: List of frame ball detection dictionaries [{1: {'bbox': ...}}, ...].
            limit: Maximum consecutive missing frames to interpolate.

        Returns:
            List of interpolated ball dictionaries.
        """
        if not ball_positions:
            return []

        extracted_boxes = []
        for frame_ball in ball_positions:
            if 1 in frame_ball and "bbox" in frame_ball[1]:
                extracted_boxes.append(frame_ball[1]["bbox"])
            else:
                extracted_boxes.append([None, None, None, None])

        df_boxes = pd.DataFrame(extracted_boxes, columns=["x1", "y1", "x2", "y2"])

        # Interpolate bounded gaps
        df_boxes = df_boxes.interpolate(limit=limit)
        df_boxes = df_boxes.bfill(limit=limit)

        interpolated_positions = []
        for row in df_boxes.to_numpy():
            if pd.isna(row[0]):
                interpolated_positions.append({})
            else:
                interpolated_positions.append({1: {"bbox": row.tolist()}})

        return interpolated_positions
