"""
Unit tests for system asset and pre-flight validation utilities.
"""

import pytest

from football_cv.config import load_config
from football_cv.exceptions import ValidationError
from football_cv.utils.validation import (
    run_preflight_checks,
    validate_device,
    validate_model_path,
    validate_video_path,
)


class TestAssetValidation:
    def test_validate_existing_model_path(self):
        info = validate_model_path("models/best.pt")
        assert info["size_bytes"] > 0
        assert info["size_mb"] > 10.0
        assert "models" in info["path"]

    def test_validate_missing_model_path_raises(self):
        with pytest.raises(ValidationError, match="Model checkpoint not found"):
            validate_model_path("models/non_existent.pt")

    def test_validate_empty_model_path_raises(self, tmp_path):
        empty_model = tmp_path / "empty.pt"
        empty_model.touch()
        with pytest.raises(ValidationError, match="empty \\(0 bytes\\)"):
            validate_model_path(empty_model)

    def test_validate_existing_video_path(self):
        info = validate_video_path("input_videos/08fd33_4.mp4", check_readable=True)
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["fps"] == 25.0
        assert info["frame_count"] == 750
        assert info["duration_seconds"] == 30.0

    def test_validate_missing_video_path_raises(self):
        with pytest.raises(ValidationError, match="Input video file not found"):
            validate_video_path("input_videos/missing.mp4")

    def test_validate_corrupted_video_raises(self, tmp_path):
        corrupt_vid = tmp_path / "corrupt.mp4"
        corrupt_vid.write_bytes(b"not a video stream")
        with pytest.raises(ValidationError, match="failed to open video file"):
            validate_video_path(corrupt_vid)


class TestDeviceValidation:
    def test_validate_cpu_device(self):
        assert validate_device("cpu") == "cpu"
        assert validate_device("CPU ") == "cpu"

    def test_validate_auto_device(self):
        device = validate_device("auto")
        assert device in ("cpu", "cuda")


class TestPreflightRunner:
    def test_preflight_checks_on_default_config(self):
        config = load_config("configs/default.yaml")
        results = run_preflight_checks(config)
        assert results["status"] == "PASS"
        assert results["model"]["size_mb"] > 0
        assert results["video"]["width"] == 1920
        assert results["output_dirs"]["writable"] is True
