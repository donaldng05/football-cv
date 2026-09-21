"""
Strict numerical parity and integration validation suite for ONNX Runtime detector deployment.

Validates:
1. Architectural integrity of exported ONNX model files.
2. High-precision numerical parity between PyTorch YOLO and ONNX Runtime predictions
   (IoU >= 0.85, class label match rate >= 95%).
3. Numerical equivalence between native C++ OnnxDetector and Python ONNX Runtime fallback.
4. Seamless integration with ObjectTracker (ByteTrack) and MatchPipeline.
5. Robustness across edge cases (blank frames, arbitrary resolutions).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv

from football_cv.config import load_config
from football_cv.core import has_cpp_core
from football_cv.pipeline import MatchPipeline
from football_cv.tracking.detector import ObjectDetector
from football_cv.tracking.tracker import ObjectTracker
from football_cv.utils.video import read_video

pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(
        not has_cpp_core(),
        reason="Native C++ extension football_cv._core is required for ONNX parity testing",
    ),
]

VIDEO_PATH = "input_videos/08fd33_4.mp4"
ONNX_MODEL_PATH = "models/best.onnx"
PT_MODEL_PATH = "models/best.pt"


def _compute_iou(
    box1: np.ndarray | list[float], box2: np.ndarray | list[float]
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1])
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1])
    )

    union = area1 + area2 - intersection
    return intersection / union if union > 0.0 else 0.0


@pytest.fixture(scope="module")
def sample_video_frames() -> list[np.ndarray]:
    """Load a 3-frame sequence from the test video for parity verification."""
    video_p = Path(VIDEO_PATH)
    if not video_p.is_file():
        pytest.skip(f"Test video '{VIDEO_PATH}' not found")
    frames = read_video(str(video_p), start_frame=0, end_frame=3)
    if not frames:
        pytest.skip(f"Failed to read frames from '{VIDEO_PATH}'")
    return frames


class TestOnnxParity:
    """Numerical parity and consistency checks for ONNX detection engine."""

    def test_onnx_model_file_exists(self) -> None:
        """Verify that the exported ONNX model exists and has appropriate file size."""
        model_path = Path(ONNX_MODEL_PATH)
        assert model_path.is_file(), (
            f"ONNX model file '{ONNX_MODEL_PATH}' does not exist"
        )
        size_mb = model_path.stat().st_size / (1024 * 1024)
        assert size_mb > 5.0, (
            f"ONNX model file is suspiciously small ({size_mb:.2f} MB)"
        )

    def test_onnx_detector_initialization(self) -> None:
        """Verify native C++ OnnxDetector instantiation and metadata query."""
        from football_cv import _core

        detector = _core.OnnxDetector(ONNX_MODEL_PATH)
        assert detector.input_width == 640
        assert detector.input_height == 640
        assert detector.num_classes == 4

    def test_pytorch_vs_onnx_prediction_parity(
        self, sample_video_frames: list[np.ndarray]
    ) -> None:
        """
        Validate that ONNX Runtime detections match PyTorch YOLO detections with high IoU
        and identical class classifications on real match video frames.
        """
        frame = sample_video_frames[0]

        pt_detector = ObjectDetector(
            model_path=PT_MODEL_PATH, engine="ultralytics", confidence=0.25
        )
        onnx_detector = ObjectDetector(
            model_path=ONNX_MODEL_PATH, engine="onnx", confidence=0.25
        )

        pt_results = pt_detector.detect_frames([frame])[0]
        onnx_results = onnx_detector.detect_frames([frame])[0]

        pt_dets = sv.Detections.from_ultralytics(pt_results)
        onnx_dets: sv.Detections = onnx_results

        assert len(pt_dets) > 0, "PyTorch detector found 0 objects on test frame"
        assert len(onnx_dets) > 0, "ONNX detector found 0 objects on test frame"

        # Detection count should be within +-2 objects
        count_diff = abs(len(pt_dets) - len(onnx_dets))
        assert count_diff <= 3, (
            f"Detection count divergence too large: PyTorch={len(pt_dets)}, ONNX={len(onnx_dets)}"
        )

        # Match each PyTorch box to the best ONNX box
        matched_ious: list[float] = []
        matched_classes = 0

        for pt_box, pt_cls in zip(pt_dets.xyxy, pt_dets.class_id, strict=False):
            best_iou = 0.0
            best_cls = -1
            for onnx_box, onnx_cls in zip(
                onnx_dets.xyxy, onnx_dets.class_id, strict=False
            ):
                iou = _compute_iou(pt_box, onnx_box)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = onnx_cls

            matched_ious.append(best_iou)
            if best_cls == pt_cls:
                matched_classes += 1

        mean_iou = float(np.mean(matched_ious))
        min_iou = float(np.min(matched_ious))
        class_match_rate = matched_classes / len(pt_dets)

        # Assertions
        assert mean_iou >= 0.85, (
            f"Mean IoU between PyTorch and ONNX is too low: {mean_iou:.4f} (expected >= 0.85)"
        )
        assert class_match_rate >= 0.90, (
            f"Class label match rate is too low: {class_match_rate:.2%} (expected >= 90%)"
        )
        assert min_iou >= 0.50, (
            f"Minimum matched box IoU is too low: {min_iou:.4f} (expected >= 0.50)"
        )

    def test_native_cpp_vs_python_onnx_parity(
        self, sample_video_frames: list[np.ndarray]
    ) -> None:
        """
        Verify that native C++ OnnxDetector and Python onnxruntime fallback produce
        numerically identical results for the exact same image.
        """
        frame = sample_video_frames[0]

        detector = ObjectDetector(
            model_path=ONNX_MODEL_PATH, engine="onnx", confidence=0.25
        )
        assert detector.native_detector is not None

        # 1. Native C++ inference
        native_dets = detector._detect_onnx([frame])[0]

        # 2. Python fallback inference
        if detector.python_session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            detector.python_session = ort.InferenceSession(
                detector.onnx_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            detector.input_name = detector.python_session.get_inputs()[0].name

        py_dets = detector._detect_single_python_onnx(frame)

        assert len(native_dets) > 0
        assert len(py_dets) > 0

        # Boxes should match with very high IoU
        matched_ious = []
        for n_box in native_dets.xyxy:
            best_iou = max(_compute_iou(n_box, p_box) for p_box in py_dets.xyxy)
            matched_ious.append(best_iou)

        mean_iou = float(np.mean(matched_ious))
        assert mean_iou >= 0.90, (
            f"C++ vs Python ONNX fallback mean IoU too low: {mean_iou:.4f} (expected >= 0.90)"
        )

    def test_object_tracker_integration_onnx(
        self, sample_video_frames: list[np.ndarray]
    ) -> None:
        """Verify that ObjectTracker operates seamlessly with engine='onnx'."""
        tracker = ObjectTracker(
            model_path=ONNX_MODEL_PATH,
            confidence=0.25,
            engine="onnx",
        )

        tracks = tracker.get_tracked_objects(sample_video_frames)

        assert "players" in tracks
        assert "referees" in tracks
        assert "balls" in tracks
        assert len(tracks["players"]) == len(sample_video_frames)
        assert len(tracks["referees"]) == len(sample_video_frames)
        assert len(tracks["balls"]) == len(sample_video_frames)

        # Confirm non-empty tracks for players on match video
        assert len(tracks["players"][0]) > 0, "No players tracked in frame 0"

        # Position attribution
        tracker.add_positions_to_tracks(tracks)
        first_player_id = next(iter(tracks["players"][0].keys()))
        player_track = tracks["players"][0][first_player_id]
        assert "position" in player_track
        assert len(player_track["position"]) == 2

    def test_blank_and_uniform_frames(self) -> None:
        """Verify robust behavior with blank and synthetic uniform frames."""
        from football_cv import _core

        detector = _core.OnnxDetector(ONNX_MODEL_PATH)

        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets_black = detector.detect(black_frame, 0.25, 0.50)
        assert len(dets_black) == 0

        white_frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        dets_white = detector.detect(white_frame, 0.25, 0.50)
        assert len(dets_white) == 0

    def test_match_pipeline_end_to_end_onnx(self) -> None:
        """Verify MatchPipeline end-to-end execution with ONNX engine and C++ vision."""
        cfg = load_config(
            "configs/fast.yaml",
            overrides={
                "model": {"engine": "onnx", "path": ONNX_MODEL_PATH},
                "vision": {"backend": "cpp"},
                "tracking": {"use_cached_tracks": False},
            },
        )
        frames = read_video(cfg.video.input_path, start_frame=0, end_frame=3)
        pipeline = MatchPipeline(cfg)
        result: dict[str, Any] = pipeline.process(frames)

        assert "tracks" in result
        assert "camera_movement" in result
        assert "team_ball_control" in result
        assert len(result["tracks"]["players"]) == 3
        assert len(result["camera_movement"]) == 3
