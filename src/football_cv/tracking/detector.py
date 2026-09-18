"""
Object detection module leveraging Ultralytics YOLOv8.
"""

from typing import Any

import numpy as np
from ultralytics import YOLO

from ..exceptions import ModelError


class ObjectDetector:
    """YOLOv8 wrapper for batch object detection on video frames."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.10,
        batch_size: int = 20,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.batch_size = batch_size
        self.device = device

        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise ModelError(
                f"Failed to load YOLO model from {model_path}: {exc}"
            ) from exc

    def detect_frames(self, frames: list[np.ndarray]) -> list[Any]:
        """
        Run inference across a list of frames using batching.

        Args:
            frames: List of BGR images as NumPy arrays.

        Returns:
            List of Ultralytics Results objects.
        """
        if not frames:
            return []

        all_detections = []
        predict_kwargs = {"conf": self.confidence, "verbose": False}
        if self.device is not None and self.device != "auto":
            predict_kwargs["device"] = self.device

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            results = self.model.predict(batch, **predict_kwargs)
            all_detections.extend(results)

        return all_detections
