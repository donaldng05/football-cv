"""
Unit tests for vectorized YOLOv8 anchor decoding, ONNX batch inference, and pipeline profiling.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from football_cv.benchmark.micro import benchmark_detection_postprocessing
from football_cv.benchmark.profiler import (
    ProfileSummary,
    StageMetrics,
    format_profile_summary_table,
)
from football_cv.config import load_config
from football_cv.core import has_cpp_core
from football_cv.pipeline import MatchPipeline
from football_cv.tracking.detector import ObjectDetector


class TestDetectorVectorization:
    """Tests for vectorized ONNX letterboxing and anchor post-processing."""

    @pytest.fixture
    def detector(self) -> ObjectDetector:
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.confidence = 0.25
        detector.nms_threshold = 0.50
        detector.batch_size = 4
        detector.engine = "onnx"
        detector.native_detector = None
        detector.input_name = "images"

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 640, 640]
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.zeros((1, 8, 8400), dtype=np.float32)]
        detector.python_session = mock_session
        return detector

    def test_live_onnx_detector_initialization_if_available(self) -> None:
        """Verify live ONNX detector initialization when native C++ or onnxruntime is available."""
        if not has_cpp_core() and importlib.util.find_spec("onnxruntime") is None:
            pytest.skip("Neither native C++ _core nor onnxruntime is available")

        detector = ObjectDetector(
            model_path="models/best.onnx",
            engine="onnx",
            confidence=0.25,
            batch_size=4,
        )
        assert detector.engine == "onnx"

    def test_preprocess_letterbox_dimensions(self) -> None:
        """Verify letterbox output shape and normalization for 1080p frame."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tensor, pad_x, pad_y, inv_r, orig_w, orig_h = (
            ObjectDetector._preprocess_letterbox(frame, target_w=640, target_h=640)
        )

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= float(tensor.max()) <= 1.0
        assert orig_w == 1920
        assert orig_h == 1080
        assert pad_x == pytest.approx(0.0, abs=1e-2)
        assert pad_y == pytest.approx(140.0, abs=1.0)
        assert inv_r == pytest.approx(3.0, abs=1e-2)

    def test_postprocess_yolo_tensor_matches_scalar_exact(
        self, detector: ObjectDetector
    ) -> None:
        """Verify 100% bit-exact parity between scalar loop and vectorized NumPy postprocessing."""
        rng = np.random.default_rng(12345)
        pred = rng.standard_normal((8, 8400)).astype(np.float32)
        # Give positive scale to boxes
        pred[2:4, :] = np.abs(pred[2:4, :]) * 50.0 + 10.0
        pred[0:2, :] = np.abs(pred[0:2, :]) * 300.0 + 50.0
        # High confidence for select anchors
        pred[4:8, :30] = rng.uniform(0.7, 0.99, (4, 30)).astype(np.float32)

        orig_w, orig_h = 1920, 1080
        pad_x, pad_y = 0.0, 140.0
        inv_r = 3.0
        num_classes = 4

        # 1. Scalar reference implementation
        boxes_scalar, scores_scalar, classes_scalar = [], [], []
        for i in range(pred.shape[1]):
            class_scores = pred[4 : 4 + num_classes, i]
            best_cls = int(np.argmax(class_scores))
            max_score = float(class_scores[best_cls])
            if max_score >= detector.confidence:
                cx, cy, bw, bh = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
                x1 = max(0.0, min(float(orig_w), (cx - bw * 0.5 - pad_x) * inv_r))
                y1 = max(0.0, min(float(orig_h), (cy - bh * 0.5 - pad_y) * inv_r))
                x2 = max(0.0, min(float(orig_w), (cx + bw * 0.5 - pad_x) * inv_r))
                y2 = max(0.0, min(float(orig_h), (cy + bh * 0.5 - pad_y) * inv_r))
                if x2 > x1 and y2 > y1:
                    boxes_scalar.append([x1, y1, x2, y2])
                    scores_scalar.append(max_score)
                    classes_scalar.append(best_cls)

        raw_scalar = sv.Detections(
            xyxy=np.array(boxes_scalar, dtype=np.float32),
            confidence=np.array(scores_scalar, dtype=np.float32),
            class_id=np.array(classes_scalar, dtype=int),
        )
        dets_scalar = raw_scalar.with_nms(threshold=detector.nms_threshold)

        # 2. Vectorized implementation under test
        dets_vectorized = detector._postprocess_yolo_tensor(
            pred, orig_w, orig_h, pad_x, pad_y, inv_r
        )

        assert len(dets_scalar) == len(dets_vectorized)
        assert len(dets_vectorized) > 0
        np.testing.assert_allclose(
            dets_scalar.xyxy, dets_vectorized.xyxy, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            dets_scalar.confidence, dets_vectorized.confidence, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(dets_scalar.class_id, dets_vectorized.class_id)

    def test_postprocess_yolo_tensor_empty_on_zero_conf(
        self, detector: ObjectDetector
    ) -> None:
        """Verify that tensors with zero confidence produce empty detections without crash."""
        pred = np.zeros((8, 8400), dtype=np.float32)
        dets = detector._postprocess_yolo_tensor(
            pred, orig_w=1920, orig_h=1080, pad_x=0.0, pad_y=0.0, inv_r=1.0
        )
        assert len(dets) == 0
        assert dets.xyxy.shape == (0, 4)
        assert dets.confidence.shape == (0,)
        assert dets.class_id.shape == (0,)

    def test_batch_detection_across_chunk_frames(
        self, detector: ObjectDetector
    ) -> None:
        """Verify detect_frames handles multi-frame batch slicing cleanly."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(7)]
        results = detector.detect_frames(frames)
        assert len(results) == 7
        for r in results:
            assert isinstance(r, sv.Detections)
            assert len(r) == 0

    def test_microbenchmark_detection_postprocessing(self) -> None:
        """Verify that microbenchmark executes and records substantial vectorization speedup."""
        results = benchmark_detection_postprocessing(iterations=5)
        assert len(results) == 1
        res = results[0]
        assert res.name == "onnx_yolo_anchor_decoding"
        assert res.speedup > 1.0, f"Expected speedup > 1.0, got {res.speedup}"
        assert res.python_latency_us > res.cpp_latency_us


class TestPipelineProfiling:
    """Tests for integrated stage-by-stage pipeline profiling."""

    def test_format_profile_summary_table(self) -> None:
        """Verify ASCII summary table rendering."""
        summary = ProfileSummary(
            num_frames=100,
            total_duration_seconds=1.2345,
            throughput_fps=81.0,
            stages={
                "detection_and_tracking": StageMetrics(
                    stage_name="detection_and_tracking",
                    total_seconds=0.7407,
                    ms_per_frame=7.41,
                    percent_of_total=60.0,
                ),
                "rendering": StageMetrics(
                    stage_name="rendering",
                    total_seconds=0.4938,
                    ms_per_frame=4.94,
                    percent_of_total=40.0,
                ),
            },
        )
        table = format_profile_summary_table(summary)
        assert "Pipeline Latency & Throughput Profile" in table
        assert "detection_and_tracking" in table
        assert "rendering" in table
        assert "81.00 FPS" in table

    def test_pipeline_run_with_profiling_populates_summary(
        self, tmp_path: Path
    ) -> None:
        """Verify that running pipeline with profile=True attaches profile metrics to results."""
        config = load_config("configs/fast.yaml")
        config.video.start_frame = 0
        config.video.end_frame = 4
        config.tracking.use_cached_tracks = False
        config.analytics.export_dir = str(tmp_path / "analytics")

        pipeline = MatchPipeline(config)
        results = pipeline.run(profile=True)

        assert "profile_summary" in results
        assert "profile_summary_obj" in results
        summary = results["profile_summary"]
        assert summary["num_frames"] == 4
        assert summary["throughput_fps"] > 0.0

        stages = summary["stages"]
        expected_stages = {
            "detection_and_tracking",
            "camera_motion",
            "perspective_transform",
            "speed_distance",
            "team_assignment",
            "possession_assignment",
            "rendering",
        }
        for stage in expected_stages:
            assert stage in stages, f"Expected stage '{stage}' in profile summary"

    def test_pipeline_streaming_run_with_profiling(self, tmp_path: Path) -> None:
        """Verify that streaming execution with profile=True collects all stage metrics."""
        config = load_config("configs/fast.yaml")
        config.video.start_frame = 0
        config.video.end_frame = 8
        config.streaming.enabled = True
        config.streaming.chunk_size = 4
        config.analytics.export_dir = str(tmp_path / "analytics_streaming")

        pipeline = MatchPipeline(config)
        results = pipeline.run(profile=True)

        assert "profile_summary" in results
        summary = results["profile_summary"]
        assert summary["num_frames"] == 8
        assert summary["throughput_fps"] > 0.0
        assert "detection_and_tracking" in summary["stages"]
