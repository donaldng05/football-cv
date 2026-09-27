"""
Kinematic ball tracker with Kalman-style constant-velocity state estimation and displacement gating.
"""

from typing import Any

import numpy as np


class BallTracker:
    """
    Dedicated ball tracker designed for sports dynamics:
    - Filters out teleportation false positives (referee socks, boots, sideline noise).
    - Prioritizes high-confidence detections near predicted trajectory.
    - Smooths bounding box coordinates and predicts ball position during brief dropouts.
    """

    def __init__(
        self,
        max_displacement_pixels: float = 200.0,
        min_confidence: float = 0.15,
        max_lost_frames: int = 10,
        alpha_pos: float = 0.85,
        alpha_vel: float = 0.60,
    ):
        self.max_displacement_pixels = max_displacement_pixels
        self.min_confidence = min_confidence
        self.max_lost_frames = max_lost_frames
        self.alpha_pos = alpha_pos
        self.alpha_vel = alpha_vel

        # Internal state: position (cx, cy), velocity (vx, vy), bbox dimensions (w, h)
        self.position: tuple[float, float] | None = None
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.last_bbox_dims: tuple[float, float] = (20.0, 20.0)
        self.last_confidence: float = 1.0
        self.lost_frames: int = 0

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_bbox_dims = (20.0, 20.0)
        self.last_confidence = 1.0
        self.lost_frames = 0

    def predict(self) -> tuple[float, float] | None:
        """Predict ball position in current frame based on constant velocity model."""
        if self.position is None:
            return None
        return (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1],
        )

    @staticmethod
    def _get_bbox_center(bbox: list[float]) -> tuple[float, float]:
        return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _get_bbox_dims(bbox: list[float]) -> tuple[float, float]:
        return (float(abs(bbox[2] - bbox[0])), float(abs(bbox[3] - bbox[1])))

    def update(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Update ball tracker with frame candidate detections.

        Args:
            candidates: List of candidate detections with 'bbox' [x1, y1, x2, y2]
                        and optional 'confidence'.

        Returns:
            Dictionary {'bbox': [...], 'confidence': float} for tracked ball, or None.
        """
        valid_candidates: list[dict[str, Any]] = []
        for cand in candidates:
            bbox = cand.get("bbox", [])
            if len(bbox) >= 4:
                conf = float(cand.get("confidence", 1.0))
                if conf >= self.min_confidence:
                    valid_candidates.append({"bbox": bbox, "confidence": conf})

        # 1. Tracker Uninitialized or Lost Beyond Horizon
        if self.position is None or self.lost_frames >= self.max_lost_frames:
            if not valid_candidates:
                self.lost_frames += 1
                return None

            # Pick highest confidence candidate to initialize
            best_cand = max(valid_candidates, key=lambda c: c["confidence"])
            cx, cy = self._get_bbox_center(best_cand["bbox"])
            self.position = (cx, cy)
            self.velocity = (0.0, 0.0)
            self.last_bbox_dims = self._get_bbox_dims(best_cand["bbox"])
            self.last_confidence = best_cand["confidence"]
            self.lost_frames = 0
            return dict(best_cand)

        # 2. Tracker Active: Predict position and apply kinematic gating
        pred_x, pred_y = self.predict()  # type: ignore[misc]

        gated_candidates: list[tuple[float, dict[str, Any], float, float]] = []
        for cand in valid_candidates:
            cx, cy = self._get_bbox_center(cand["bbox"])
            dist = float(np.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2))

            if dist <= self.max_displacement_pixels:
                # Combined scoring: high confidence + proximity to prediction
                score = cand["confidence"] - 0.35 * (
                    dist / self.max_displacement_pixels
                )
                gated_candidates.append((score, cand, cx, cy))

        if gated_candidates:
            # Pick candidate with highest score
            gated_candidates.sort(key=lambda item: item[0], reverse=True)
            _, best_cand, meas_x, meas_y = gated_candidates[0]

            # Measurement velocity update
            obs_vx = meas_x - self.position[0]
            obs_vy = meas_y - self.position[1]

            new_vx = (1.0 - self.alpha_vel) * self.velocity[0] + self.alpha_vel * obs_vx
            new_vy = (1.0 - self.alpha_vel) * self.velocity[1] + self.alpha_vel * obs_vy

            # Position smoothing update
            new_x = (1.0 - self.alpha_pos) * pred_x + self.alpha_pos * meas_x
            new_y = (1.0 - self.alpha_pos) * pred_y + self.alpha_pos * meas_y

            self.position = (new_x, new_y)
            self.velocity = (new_vx, new_vy)
            self.last_bbox_dims = self._get_bbox_dims(best_cand["bbox"])
            self.last_confidence = best_cand["confidence"]
            self.lost_frames = 0

            return dict(best_cand)

        # 3. No candidate passed gating: Coast with predicted trajectory for brief dropouts
        self.lost_frames += 1
        if self.lost_frames <= 2:
            # Construct coasted bounding box centered at prediction
            self.position = (pred_x, pred_y)
            half_w = self.last_bbox_dims[0] / 2.0
            half_h = self.last_bbox_dims[1] / 2.0
            coasted_bbox = [
                pred_x - half_w,
                pred_y - half_h,
                pred_x + half_w,
                pred_y + half_h,
            ]
            coasted_conf = max(0.10, self.last_confidence * 0.70)
            return {"bbox": coasted_bbox, "confidence": coasted_conf}

        return None
