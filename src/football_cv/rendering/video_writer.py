"""
Annotated video rendering and streaming writer.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..exceptions import VideoProcessingError
from .annotations import FrameAnnotator


class IncrementalVideoWriter:
    """
    Encodes and writes video frames incrementally without retaining history in RAM.
    Supports context manager (`with IncrementalVideoWriter(...) as writer:`).
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 25.0,
        codec: str = "XVID",
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer: cv2.VideoWriter | None = None
        self._is_opened = False

    def __enter__(self) -> "IncrementalVideoWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _initialize_writer(self, width: int, height: int) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (width, height)
        )

        if not self.writer.isOpened():
            # Fallback to mp4v if requested codec is unsupported
            fallback_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                str(self.output_path), fallback_fourcc, self.fps, (width, height)
            )
            if not self.writer.isOpened():
                raise VideoProcessingError(
                    f"Failed to open VideoWriter for {self.output_path}"
                )

        self._is_opened = True

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video stream."""
        if frame is None or frame.size == 0:
            raise VideoProcessingError("Cannot write empty or None frame")

        if not self._is_opened or self.writer is None:
            height, width = frame.shape[:2]
            self._initialize_writer(width, height)

        self.writer.write(frame)

    def write_chunk(self, frames: Sequence[np.ndarray]) -> None:
        """Write a sequence or chunk of frames to the video stream."""
        if not frames:
            return

        for frame in frames:
            self.write_frame(frame)

    def close(self) -> None:
        """Release native VideoWriter resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self._is_opened = False


class AnnotatedVideoWriter(IncrementalVideoWriter):
    """
    Backward-compatible wrapper for AnnotatedVideoWriter that supports both
    batch write_frames and incremental/streaming writing.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 25.0,
        codec: str = "XVID",
    ):
        super().__init__(output_path=output_path, fps=fps, codec=codec)
        self.annotator = FrameAnnotator()

    def write_frames(self, frames: list[np.ndarray]) -> None:
        """Write a complete list of frames to the output file."""
        if not frames:
            raise VideoProcessingError("Cannot write empty frames list")

        try:
            self.write_chunk(frames)
        finally:
            self.close()
