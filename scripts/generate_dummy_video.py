"""
Generate a synthetic match video for CI and automated testing environments.
Ensures pre-flight asset checks and smoke tests pass when external video
assets are excluded from git version control.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_dummy_video(
    output_path: str | Path = "input_videos/08fd33_4.mp4",
    num_frames: int = 750,
    width: int = 1920,
    height: int = 1080,
    fps: float = 25.0,
) -> Path:
    """
    Generate a synthetic video container matching expected dimensions and FPS.

    Args:
        output_path: Destination path for the generated video file.
        num_frames: Total number of frames to write (default 750 = 30s @ 25fps).
        width: Video width in pixels (default 1920).
        height: Video height in pixels (default 1080).
        fps: Frames per second (default 25.0).

    Returns:
        Path to the verified video file.
    """
    target = Path(output_path)
    if target.exists() and target.stat().st_size > 0:
        print(
            f"[INFO] Video asset already exists: {target} ({target.stat().st_size} bytes)"
        )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))

    # Generate a lightweight green pitch backdrop frame
    frame = np.full((height, width, 3), fill_value=(40, 120, 40), dtype=np.uint8)

    for _ in range(num_frames):
        writer.write(frame)

    writer.release()
    print(
        f"[INFO] Generated synthetic video asset: {target} ({num_frames} frames, {width}x{height} @ {fps} FPS)"
    )
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic match video for testing"
    )
    parser.add_argument(
        "--output", default="input_videos/08fd33_4.mp4", help="Output video path"
    )
    parser.add_argument("--frames", type=int, default=750, help="Number of frames")
    args = parser.parse_args()

    generate_dummy_video(output_path=args.output, num_frames=args.frames)
