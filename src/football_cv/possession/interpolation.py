"""
Ball position interpolation for bridging missing detection frames.
"""

from typing import Any

import numpy as np
import pandas as pd


class BallInterpolator:
    """
    Interpolates missing ball detections using physics-aware temporal interpolation.
    - Bridges detection dropouts up to a configurable horizon (e.g. 15 frames).
    - Checks implied velocity to avoid bridging across disconnected false positives.
    - Supports quadratic / polynomial arc interpolation for curved and aerial trajectories.
    """

    @staticmethod
    def interpolate_ball_positions(
        ball_positions: list[dict[int, dict[str, Any]]],
        limit: int = 15,
        method: str = "linear",
        max_displacement_per_frame: float = 250.0,
    ) -> list[dict[int, dict[str, Any]]]:
        """
        Interpolate missing ball coordinates across consecutive frames.

        Args:
            ball_positions: List of frame ball detection dictionaries [{1: {'bbox': ...}}, ...].
            limit: Maximum consecutive missing frames to interpolate.
            method: Interpolation method ('linear' or 'quadratic').
            max_displacement_per_frame: Maximum plausible pixel displacement per frame.

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

        df_boxes = pd.DataFrame(
            extracted_boxes, columns=["x1", "y1", "x2", "y2"], dtype=float
        )

        # If no ball detections exist across all frames, return original empty frames
        if df_boxes.isna().all().all():
            return ball_positions

        # 1. Physics-based velocity gating across gaps
        # Check gaps between valid observations; if implied velocity is impossible, record to mask out
        invalid_gaps: list[tuple[int, int]] = []
        valid_indices = df_boxes.dropna(how="all").index.tolist()
        if len(valid_indices) >= 2:
            for k in range(len(valid_indices) - 1):
                idx_start = valid_indices[k]
                idx_end = valid_indices[k + 1]
                gap_len = idx_end - idx_start

                if gap_len > 1:
                    box_s = df_boxes.iloc[idx_start]
                    box_e = df_boxes.iloc[idx_end]
                    cs_x = (box_s["x1"] + box_s["x2"]) / 2.0
                    cs_y = (box_s["y1"] + box_s["y2"]) / 2.0
                    ce_x = (box_e["x1"] + box_e["x2"]) / 2.0
                    ce_y = (box_e["y1"] + box_e["y2"]) / 2.0

                    jump_dist = float(np.sqrt((ce_x - cs_x) ** 2 + (ce_y - cs_y) ** 2))
                    if jump_dist > max_displacement_per_frame * gap_len:
                        invalid_gaps.append((idx_start + 1, idx_end))

        # 2. Trajectory interpolation
        interp_method = method.lower()
        num_valid = len(valid_indices)

        if interp_method == "quadratic" and num_valid >= 3:
            try:
                df_boxes = df_boxes.interpolate(method="quadratic", limit=limit)
            except Exception:
                df_boxes = df_boxes.interpolate(method="linear", limit=limit)
        else:
            df_boxes = df_boxes.interpolate(method="linear", limit=limit)

        # Invalidate impossible teleportation gaps
        for start_idx, end_idx in invalid_gaps:
            df_boxes.iloc[start_idx:end_idx] = np.nan

        interpolated_positions = []
        for idx, row in enumerate(df_boxes.to_numpy()):
            if pd.isna(row[0]):
                interpolated_positions.append({})
            else:
                orig_info = (
                    ball_positions[idx].get(1, {}) if idx < len(ball_positions) else {}
                )
                entry = dict(orig_info)
                entry["bbox"] = [float(c) for c in row.tolist()]
                interpolated_positions.append({1: entry})

        return interpolated_positions
