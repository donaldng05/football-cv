"""
Unit tests for vision backend resolution, configuration schema, and factory facades.
"""

import numpy as np
import pytest

from football_cv.config import VisionConfig
from football_cv.core import (
    get_camera_motion_estimator,
    get_perspective_transformer,
    has_cpp_core,
    resolve_backend,
)
from football_cv.exceptions import ConfigurationError
from football_cv.perspective.transformer import PerspectiveTransformer


class TestVisionConfig:
    """Test suite for VisionConfig schema and validation."""

    def test_vision_config_defaults(self):
        config = VisionConfig()
        assert config.backend == "python"
        config.validate()

    @pytest.mark.parametrize("valid_backend", ["python", "cpp", "PYTHON", "Cpp"])
    def test_vision_config_valid_backends(self, valid_backend: str):
        config = VisionConfig(backend=valid_backend)
        config.validate()

    @pytest.mark.parametrize(
        "invalid_backend", ["cuda", "rust", "opencl", "", "native"]
    )
    def test_vision_config_invalid_backends_raise(self, invalid_backend: str):
        with pytest.raises(ConfigurationError, match="Invalid vision backend"):
            VisionConfig(backend=invalid_backend).validate()


class TestBackendResolution:
    """Test suite for runtime backend resolution and fallback logic."""

    def test_has_cpp_core_returns_bool(self):
        result = has_cpp_core()
        assert isinstance(result, bool)

    def test_resolve_backend_default(self):
        assert resolve_backend(None) == "python"

    def test_resolve_backend_explicit_python(self):
        assert resolve_backend("python") == "python"
        assert resolve_backend("PYTHON") == "python"

    def test_resolve_backend_invalid_raises(self):
        with pytest.raises(ConfigurationError, match="Unsupported vision backend"):
            resolve_backend("invalid_backend")

    def test_resolve_backend_cpp_fallback_when_unavailable(self, caplog):
        if not has_cpp_core():
            resolved = resolve_backend("cpp", strict=False)
            assert resolved == "python"
            assert any(
                "Falling back to pure-Python" in record.message
                for record in caplog.records
            )
        else:
            assert resolve_backend("cpp") == "cpp"

    def test_resolve_backend_cpp_strict_raises_when_unavailable(self):
        if not has_cpp_core():
            with pytest.raises(
                ConfigurationError,
                match=r"native module 'football_cv\._core' is not compiled",
            ):
                resolve_backend("cpp", strict=True)
        else:
            assert resolve_backend("cpp", strict=True) == "cpp"


class TestBackendFactories:
    """Test suite for backend factory functions."""

    def test_get_perspective_transformer_python(self):
        transformer = get_perspective_transformer(backend="python")
        assert isinstance(transformer, PerspectiveTransformer)
        assert transformer.court_width == 68.0

        # Verify transformation contract
        point = np.array([500.0, 500.0])
        transformed = transformer.transform_point(point)
        assert transformed is not None
        assert transformed.shape == (1, 2)

    def test_get_perspective_transformer_cpp_fallback(self):
        if not has_cpp_core():
            transformer = get_perspective_transformer(backend="cpp", strict=False)
            assert isinstance(transformer, PerspectiveTransformer)
        else:
            transformer = get_perspective_transformer(backend="cpp")
            assert transformer is not None

    def test_get_perspective_transformer_cpp_strict(self):
        if not has_cpp_core():
            with pytest.raises(
                ConfigurationError,
                match=r"native module 'football_cv\._core' is not compiled",
            ):
                get_perspective_transformer(backend="cpp", strict=True)

    def test_get_camera_motion_estimator_python(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        estimator = get_camera_motion_estimator(dummy_frame, backend="python")
        assert estimator is not None
        assert estimator.minimum_distance == 5.0

    def test_get_camera_motion_estimator_cpp_fallback(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        if not has_cpp_core():
            estimator = get_camera_motion_estimator(
                dummy_frame, backend="cpp", strict=False
            )
            assert estimator is not None
        else:
            estimator = get_camera_motion_estimator(dummy_frame, backend="cpp")
            assert estimator is not None

    def test_get_camera_motion_estimator_cpp_strict(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        if not has_cpp_core():
            with pytest.raises(
                ConfigurationError,
                match=r"native module 'football_cv\._core' is not compiled",
            ):
                get_camera_motion_estimator(dummy_frame, backend="cpp", strict=True)
