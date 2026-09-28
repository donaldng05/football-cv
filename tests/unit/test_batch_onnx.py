"""
Unit tests for dynamic ONNX batching, multi-frame letterbox vectorization, and C++ adapter parity.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from football_cv.core import get_perspective_transformer, has_cpp_core
from football_cv.perspective.transformer import PerspectiveTransformer
from football_cv.tracking.detector import ObjectDetector


class TestBatchPreprocessing:
    """Tests for multi-frame contiguous letterbox vectorization."""

    def test_preprocess_batch_letterbox_matches_single_exact(self) -> None:
        rng = np.random.default_rng(42)
        frames = [
            rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8),
            rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        ]

        batch_tensor, transforms = ObjectDetector._preprocess_batch_letterbox(
            frames, target_w=640, target_h=640
        )

        assert batch_tensor.shape == (3, 3, 640, 640)
        assert len(transforms) == 3

        for i, frame in enumerate(frames):
            single_tensor, pad_x, pad_y, inv_r, orig_w, orig_h = (
                ObjectDetector._preprocess_letterbox(frame, target_w=640, target_h=640)
            )
            np.testing.assert_allclose(batch_tensor[i], single_tensor[0], atol=1e-5)
            assert transforms[i] == (pad_x, pad_y, inv_r, orig_w, orig_h)


class TestDynamicBatchDetection:
    """Tests for true dynamic ONNX batch dispatch vs fallback paths."""

    def test_dynamic_batch_executes_single_batched_session_run(self) -> None:
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.confidence = 0.25
        detector.nms_threshold = 0.50
        detector.batch_size = 4
        detector.engine = "onnx"
        detector.native_detector = None
        detector.input_name = "images"

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = ["batch", 3, 640, 640]  # Dynamic batch dimension!
        mock_session.get_inputs.return_value = [mock_input]

        # 4-frame batch output prediction: shape (4, 8, 8400)
        mock_session.run.return_value = [np.zeros((4, 8, 8400), dtype=np.float32)]
        detector.python_session = mock_session

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        detections = detector._detect_onnx(frames)

        assert len(detections) == 4
        # Single session.run() execution for the whole batch
        assert mock_session.run.call_count == 1
        called_args, called_kwargs = mock_session.run.call_args
        called_feed = (
            called_args[1]
            if len(called_args) > 1
            else called_kwargs.get("feed_dict", {})
        )
        if "images" in called_feed:
            assert called_feed["images"].shape == (4, 3, 640, 640)

    def test_static_batch_1_falls_back_to_sequential_safely(self) -> None:
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.confidence = 0.25
        detector.nms_threshold = 0.50
        detector.batch_size = 4
        detector.engine = "onnx"
        detector.native_detector = None
        detector.input_name = "images"

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 640, 640]  # Frozen batch 1!
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 8, 8400), dtype=np.float32)]
        detector.python_session = mock_session

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
        detections = detector._detect_onnx(frames)

        assert len(detections) == 3
        # Must execute 3 individual single-frame calls without error
        assert mock_session.run.call_count == 3

    def test_static_batch_matching_executes_batch(self) -> None:
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.confidence = 0.25
        detector.nms_threshold = 0.50
        detector.batch_size = 2
        detector.engine = "onnx"
        detector.native_detector = None
        detector.input_name = "images"

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = [2, 3, 640, 640]  # Fixed batch 2 matching batch_size!
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((2, 8, 8400), dtype=np.float32)]
        detector.python_session = mock_session

        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
        detections = detector._detect_onnx(frames)

        assert len(detections) == 2
        assert mock_session.run.call_count == 1


class TestPerspectiveBatchVectorization:
    """Tests for vectorized batch transformation across Python and C++ adapters."""

    @pytest.fixture
    def pixel_vertices(self) -> list[list[float]]:
        return [
            [110.0, 1035.0],
            [265.0, 275.0],
            [910.0, 260.0],
            [1640.0, 915.0],
        ]

    def test_python_and_cpp_transform_points_batch_parity(
        self, pixel_vertices: list[list[float]]
    ) -> None:
        py_transformer = PerspectiveTransformer(pixel_vertices)
        pts = np.array(
            [
                [200.0, 500.0],
                [500.0, 400.0],
                [800.0, 300.0],
                [100.0, 100.0],
            ],
            dtype=np.float32,
        )

        py_batch = py_transformer.transform_points_batch(pts)
        assert py_batch.shape == (4, 2)

        if has_cpp_core():
            cpp_transformer = get_perspective_transformer(pixel_vertices, backend="cpp")
            cpp_batch = cpp_transformer.transform_points_batch(pts)
            assert cpp_batch.shape == (4, 2)
            np.testing.assert_allclose(py_batch, cpp_batch, atol=1e-5)
