"""
Annotated video rendering and streaming writer.
"""

from pathlib import Path

import cv2
import numpy as np

from ..exceptions import VideoProcessingError
from .annotations import FrameAnnotator


class AnnotatedVideoWriter:
    """Encodes and saves annotated match video frames."""

    def __init__(self, output_path: str | Path, fps: float = 25.0, codec: str = "XVID"):
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer: cv2.VideoWriter | None = None
        self.annotator = FrameAnnotator()

    def write_frames(self, frames: list[np.ndarray]) -> None:
        """Write a complete list of frames to the output file."""
        if not frames:
            raise VideoProcessingError("Cannot write empty frames list")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (width, height)
        )

        if not writer.isOpened():
            # Fallback to mp4v if XVID is unsupported
            fallback_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self.output_path), fallback_fourcc, self.fps, (width, height)
            )
            if not writer.isOpened():
                raise VideoProcessingError(
                    f"Failed to open VideoWriter for {self.output_path}"
                )

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
