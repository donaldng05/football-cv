"""
Object detection module leveraging Ultralytics YOLOv8 and C++ ONNX Runtime.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from ..core import has_cpp_core
from ..exceptions import ModelError

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

if has_cpp_core():
    from .. import _core as native_core
else:
    native_core = None


class ObjectDetector:
    """
    Object detector wrapper supporting Ultralytics PyTorch and high-performance ONNX Runtime backends.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.10,
        batch_size: int = 20,
        device: str | None = None,
        engine: str = "ultralytics",
        nms_threshold: float = 0.50,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.batch_size = batch_size
        self.device = device
        self.engine = engine.lower()
        self.nms_threshold = nms_threshold
        self.names = dict(DEFAULT_CLASS_NAMES)

        if self.engine == "onnx":
            self._init_onnx_engine()
        else:
            self._init_ultralytics_engine()

    def _init_ultralytics_engine(self) -> None:
        """Initialize Ultralytics YOLO PyTorch model."""
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.names = dict(self.model.names)
            logger.info(
                f"Initialized Ultralytics YOLO detector (conf={self.confidence}, device='{self.device}')"
            )
        except Exception as exc:
            raise ModelError(
                f"Failed to load YOLO model from {self.model_path}: {exc}"
            ) from exc

    def _init_onnx_engine(self) -> None:
        """Initialize ONNX Runtime detector with native C++ or Python fallback."""
        onnx_path = Path(self.model_path)
        if onnx_path.suffix != ".onnx":
            candidate = onnx_path.with_suffix(".onnx")
            if candidate.is_file():
                onnx_path = candidate
            elif not onnx_path.is_file():
                raise ModelError(
                    f"ONNX model file not found: {onnx_path} (or candidate: {candidate})"
                )

        self.onnx_path = str(onnx_path)
        self.native_detector = None
        self.python_session = None

        # 1. Prefer compiled C++ OnnxDetector
        if native_core is not None and hasattr(native_core, "OnnxDetector"):
            try:
                self.native_detector = native_core.OnnxDetector(self.onnx_path)
                logger.info(
                    f"Initialized native C++ OnnxDetector from '{self.onnx_path}' "
                    f"({self.native_detector.input_width}x{self.native_detector.input_height}, "
                    f"{self.native_detector.num_classes} classes)"
                )
                return
            except Exception as exc:
                logger.warning(
                    f"Failed to initialize native C++ OnnxDetector: {exc}. "
                    "Falling back to Python onnxruntime engine."
                )

        # 2. Fallback to Python onnxruntime
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.python_session = ort.InferenceSession(
                self.onnx_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.python_session.get_inputs()[0].name
            logger.info(
                f"Initialized Python onnxruntime session from '{self.onnx_path}'"
            )
        except Exception as exc:
            raise ModelError(
                f"Failed to initialize Python ONNX Runtime session: {exc}"
            ) from exc

    def detect_frames(self, frames: list[np.ndarray]) -> list[Any]:
        """
        Run inference across a list of frames.

        Args:
            frames: List of BGR images as NumPy arrays.

        Returns:
            List of detections per frame (sv.Detections or Ultralytics Results).
        """
        if not frames:
            return []

        if self.engine == "onnx":
            return self._detect_onnx(frames)
        return self._detect_ultralytics(frames)

    def _detect_ultralytics(self, frames: list[np.ndarray]) -> list[Any]:
        """Inference pass using Ultralytics YOLO PyTorch wrapper."""
        all_detections = []
        predict_kwargs = {"conf": self.confidence, "verbose": False}
        if self.device is not None and self.device != "auto":
            predict_kwargs["device"] = self.device

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            results = self.model.predict(batch, **predict_kwargs)
            all_detections.extend(results)

        return all_detections

    def _detect_onnx(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """Inference pass using native C++ or Python ONNX Runtime."""
        all_detections: list[sv.Detections] = []

        if self.native_detector is not None:
            # Native C++ detector path
            for frame in frames:
                dets = self.native_detector.detect(
                    frame, self.confidence, self.nms_threshold
                )
                if not dets:
                    all_detections.append(
                        sv.Detections(
                            xyxy=np.zeros((0, 4), dtype=np.float32),
                            confidence=np.zeros((0,), dtype=np.float32),
                            class_id=np.zeros((0,), dtype=int),
                        )
                    )
                    continue

                xyxy = np.array(
                    [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in dets],
                    dtype=np.float32,
                )
                conf = np.array([d.confidence for d in dets], dtype=np.float32)
                cls_id = np.array([d.class_id for d in dets], dtype=int)
                all_detections.append(
                    sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_id)
                )
            return all_detections

        # Python onnxruntime fallback path
        for frame in frames:
            d = self._detect_single_python_onnx(frame)
            all_detections.append(d)

        return all_detections

    def _detect_single_python_onnx(self, frame: np.ndarray) -> sv.Detections:
        """Python fallback for single-frame ONNX letterboxing and inference."""
        import cv2

        h, w = frame.shape[:2]
        r = min(640.0 / w, 640.0 / h)
        unpad_w, unpad_h = round(w * r), round(h * r)
        pad_x, pad_y = (640.0 - unpad_w) * 0.5, (640.0 - unpad_h) * 0.5
        pad_x_int, pad_y_int = round(pad_x), round(pad_y)

        resized = cv2.resize(frame, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[pad_y_int : pad_y_int + unpad_h, pad_x_int : pad_x_int + unpad_w] = (
            resized
        )

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        tensor = np.expand_dims(chw, axis=0)

        outputs = self.python_session.run(None, {self.input_name: tensor})
        pred = outputs[0][0]  # shape: (8, 8400)

        boxes, scores, classes = [], [], []
        num_classes = pred.shape[0] - 4
        inv_r = 1.0 / r

        for i in range(pred.shape[1]):
            class_scores = pred[4 : 4 + num_classes, i]
            best_cls = int(np.argmax(class_scores))
            max_score = float(class_scores[best_cls])

            if max_score >= self.confidence:
                cx, cy, bw, bh = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
                x1 = max(0.0, min(float(w), (cx - bw * 0.5 - pad_x) * inv_r))
                y1 = max(0.0, min(float(h), (cy - bh * 0.5 - pad_y) * inv_r))
                x2 = max(0.0, min(float(w), (cx + bw * 0.5 - pad_x) * inv_r))
                y2 = max(0.0, min(float(h), (cy + bh * 0.5 - pad_y) * inv_r))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(max_score)
                    classes.append(best_cls)

        if not boxes:
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=int),
            )

        raw_dets = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(scores, dtype=np.float32),
            class_id=np.array(classes, dtype=int),
        )
        return raw_dets.with_nms(threshold=self.nms_threshold)
