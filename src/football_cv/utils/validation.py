"""
Pre-flight asset, device, and environment validation utilities.
"""

from pathlib import Path
from typing import Any

import cv2
import torch

from ..config import AppConfig
from ..exceptions import ValidationError


def validate_model_path(path: str | Path) -> dict[str, Any]:
    """
    Verify model checkpoint file existence and inspect basic properties.

    Args:
        path: Path to model weights file.

    Returns:
        Dictionary with model metadata.

    Raises:
        ValidationError: If the model file does not exist or is empty.
    """
    model_p = Path(path)
    if not model_p.exists():
        raise ValidationError(f"Model checkpoint not found: {model_p}")
    if not model_p.is_file():
        raise ValidationError(f"Model checkpoint path is not a file: {model_p}")

    size_bytes = model_p.stat().st_size
    if size_bytes == 0:
        raise ValidationError(f"Model checkpoint file is empty (0 bytes): {model_p}")

    return {
        "path": str(model_p),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def validate_video_path(
    path: str | Path, check_readable: bool = True
) -> dict[str, Any]:
    """
    Verify video existence and decodability via OpenCV.

    Args:
        path: Path to video file.
        check_readable: If True, attempts to read the initial frame.

    Returns:
        Dictionary with video dimensions, fps, and frame count.

    Raises:
        ValidationError: If the file is missing, empty, or cannot be decoded.
    """
    video_p = Path(path)
    if not video_p.exists():
        raise ValidationError(f"Input video file not found: {video_p}")
    if not video_p.is_file():
        raise ValidationError(f"Input video path is not a file: {video_p}")

    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise ValidationError(
            f"OpenCV failed to open video file (corrupt or unsupported codec): {video_p}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if check_readable:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise ValidationError(
                f"Video opened but first frame could not be decoded: {video_p}"
            )

    cap.release()

    duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0

    return {
        "path": str(video_p),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_sec,
    }


def validate_device(device_str: str) -> str:
    """
    Validate and normalize requested compute device.

    Args:
        device_str: "auto", "cpu", "cuda", or GPU index (e.g. "0").

    Returns:
        Resolved device string ("cpu" or "cuda" / "cuda:N").

    Raises:
        ValidationError: If CUDA was explicitly requested but is unavailable.
    """
    cuda_available = torch.cuda.is_available()

    device_normalized = device_str.lower().strip()
    if device_normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    elif device_normalized.startswith("cuda"):
        if not cuda_available:
            raise ValidationError(
                "Device 'cuda' was requested, but torch.cuda.is_available() is False"
            )
        return device_normalized
    elif device_normalized == "cpu":
        return "cpu"
    elif device_normalized.isdigit():
        if not cuda_available:
            raise ValidationError(
                f"CUDA device {device_normalized} requested, but CUDA is not available"
            )
        return f"cuda:{device_normalized}"

    return device_str


def run_preflight_checks(config: AppConfig) -> dict[str, Any]:
    """
    Run comprehensive pre-flight sanity checks against an AppConfig.

    Args:
        config: Loaded AppConfig.

    Returns:
        Dictionary summarizing validation results for model, video, device, and storage.
    """
    results: dict[str, Any] = {
        "status": "PASS",
        "model": {},
        "video": {},
        "device": {},
        "output_dirs": {},
    }

    # 1. Model Check
    model_info = validate_model_path(config.model.path)
    results["model"] = model_info

    # 2. Video Check
    video_info = validate_video_path(config.video.input_path, check_readable=True)
    results["video"] = video_info

    # 3. Device Check
    resolved_device = validate_device(config.model.device)
    device_name = (
        torch.cuda.get_device_name(0) if resolved_device.startswith("cuda") else "CPU"
    )
    results["device"] = {
        "requested": config.model.device,
        "resolved": resolved_device,
        "device_name": device_name,
    }

    # 4. Output Directories Check
    output_video_p = Path(config.video.output_path)
    output_video_p.parent.mkdir(parents=True, exist_ok=True)

    analytics_p = Path(config.analytics.export_dir)
    analytics_p.mkdir(parents=True, exist_ok=True)

    results["output_dirs"] = {
        "video_dir": str(output_video_p.parent),
        "analytics_dir": str(analytics_p),
        "writable": True,
    }

    return results
