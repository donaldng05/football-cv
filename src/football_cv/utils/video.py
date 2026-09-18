"""
Video reading, streaming, and writing utilities.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from ..exceptions import VideoProcessingError


def get_video_properties(video_path: str | Path) -> dict:
    """Extract resolution, fps, and frame count from a video file."""
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video file: {path_str}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 25.0,
        "frame_count": frame_count,
    }


def stream_video_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Generator streaming video frames one by one without loading the full video into RAM.

    Args:
        video_path: Path to video file.
        start_frame: Index of the first frame to yield (0-indexed).
        end_frame: Index of the frame to stop before (exclusive). If None, reads to end.

    Yields:
        Tuple of (frame_index, bgr_frame_array).
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise VideoProcessingError(f"Failed to open video stream: {path_str}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_idx = start_frame
    try:
        while cap.isOpened():
            if end_frame is not None and current_idx >= end_frame:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            yield current_idx, frame
            current_idx += 1
    finally:
        cap.release()


def read_video(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[np.ndarray]:
    """
    Read video frames into a list in memory (with optional slice bounds).

    Args:
        video_path: Path to video file.
        start_frame: First frame index to read.
        end_frame: Stop frame index.

    Returns:
        List of NumPy BGR image frames.
    """
    frames = []
    for _, frame in stream_video_frames(
        video_path, start_frame=start_frame, end_frame=end_frame
    ):
        frames.append(frame)
    return frames


def save_video(
    output_video_frames: Sequence[np.ndarray],
    output_video_path: str | Path,
    fps: float = 24.0,
    codec: str = "XVID",
) -> None:
    """
    Save a sequence of image frames to a video file.

    Args:
        output_video_frames: Sequence of BGR frames.
        output_video_path: Destination path.
        fps: Playback frames per second.
        codec: FourCC codec string ('XVID', 'mp4v', etc.).
    """
    if not output_video_frames:
        raise VideoProcessingError("No frames provided to save_video")

    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not out.isOpened():
        # Fallback to mp4v if XVID fails
        fallback_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fallback_fourcc, fps, (width, height))
        if not out.isOpened():
            raise VideoProcessingError(f"Could not open VideoWriter for {out_path}")

    for frame in output_video_frames:
        out.write(frame)

    out.release()
