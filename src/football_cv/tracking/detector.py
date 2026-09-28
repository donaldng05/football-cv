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
        """Inference pass using native C++ or Python ONNX Runtime with batch dispatch."""
        all_detections: list[sv.Detections] = []

        if self.native_detector is not None:
            # Native C++ detector path with batch dispatch (single GIL release per batch)
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i : i + self.batch_size]
                batch_results = self.native_detector.detect_batch(
                    batch_frames, self.confidence, self.nms_threshold
                )
                for dets in batch_results:
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
        if self.python_session is None:
            raise ModelError("ONNX Python session is not initialized")

        input_shape = self.python_session.get_inputs()[0].shape
        input_dim0 = input_shape[0] if len(input_shape) > 0 else 1
        is_dynamic_batch = (
            isinstance(input_dim0, str)
            or input_dim0 is None
            or (isinstance(input_dim0, int) and input_dim0 < 0)
        )

        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i : i + self.batch_size]
            can_batch = (is_dynamic_batch and len(batch_frames) > 1) or (
                isinstance(input_dim0, int)
                and input_dim0 == len(batch_frames)
                and len(batch_frames) > 1
            )

            if can_batch:
                # Batched dynamic inference
                batch_input, transforms = self._preprocess_batch_letterbox(batch_frames)
                outputs = self.python_session.run(None, {self.input_name: batch_input})
                preds = outputs[0]  # shape: (B, num_channels, num_anchors)

                for b_idx in range(len(batch_frames)):
                    pad_x, pad_y, inv_r, orig_w, orig_h = transforms[b_idx]
                    d = self._postprocess_yolo_tensor(
                        preds[b_idx], orig_w, orig_h, pad_x, pad_y, inv_r
                    )
                    all_detections.append(d)
            else:
                # Single-frame inference path (fixed batch-1 models)
                for frame in batch_frames:
                    d = self._detect_single_python_onnx(frame)
                    all_detections.append(d)

        return all_detections

    @classmethod
    def _preprocess_batch_letterbox(
        cls, frames: list[np.ndarray], target_w: int = 640, target_h: int = 640
    ) -> tuple[np.ndarray, list[tuple[float, float, float, int, int]]]:
        """
        Batch preprocess multiple frames into a contiguous NCHW float32 tensor [B, 3, H, W].
        """
        b = len(frames)
        batch_tensor = np.empty((b, 3, target_h, target_w), dtype=np.float32)
        transforms: list[tuple[float, float, float, int, int]] = []

        for i, frame in enumerate(frames):
            tensor, pad_x, pad_y, inv_r, orig_w, orig_h = cls._preprocess_letterbox(
                frame, target_w=target_w, target_h=target_h
            )
            batch_tensor[i] = tensor[0]
            transforms.append((pad_x, pad_y, inv_r, orig_w, orig_h))

        return batch_tensor, transforms

    @staticmethod
    def _preprocess_letterbox(
        frame: np.ndarray, target_w: int = 640, target_h: int = 640
    ) -> tuple[np.ndarray, float, float, float, int, int]:
        """
        Resize and pad frame to target resolution with aspect ratio preserved.

        Returns:
            Tuple of (NCHW float32 tensor [1, 3, H, W], pad_x, pad_y, inv_r, orig_w, orig_h).
        """
        import cv2

        h, w = frame.shape[:2]
        r = min(float(target_w) / w, float(target_h) / h)
        unpad_w, unpad_h = round(w * r), round(h * r)
        pad_x, pad_y = (
            (float(target_w) - unpad_w) * 0.5,
            (float(target_h) - unpad_h) * 0.5,
        )
        pad_x_int, pad_y_int = round(pad_x), round(pad_y)

        resized = cv2.resize(frame, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y_int : pad_y_int + unpad_h, pad_x_int : pad_x_int + unpad_w] = (
            resized
        )

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        tensor = np.expand_dims(chw, axis=0)
        return tensor, pad_x, pad_y, 1.0 / r, w, h

    def _postprocess_yolo_tensor(
        self,
        pred: np.ndarray,
        orig_w: int,
        orig_h: int,
        pad_x: float,
        pad_y: float,
        inv_r: float,
    ) -> sv.Detections:
        """
        Vectorized YOLOv8 anchor decoding and Non-Maximum Suppression.

        Args:
            pred: Array of shape (num_channels, num_anchors) e.g. (8, 8400).
            orig_w: Original frame width in pixels.
            orig_h: Original frame height in pixels.
            pad_x: Horizontal letterbox padding in pixels.
            pad_y: Vertical letterbox padding in pixels.
            inv_r: Inverse letterbox scaling ratio (1 / r).

        Returns:
            Supervision Detections object filtered by confidence and NMS.
        """
        num_classes = pred.shape[0] - 4
        class_scores = pred[4 : 4 + num_classes, :]
        best_classes = np.argmax(class_scores, axis=0)
        max_scores = class_scores[best_classes, np.arange(class_scores.shape[1])]

        mask = max_scores >= self.confidence
        if not np.any(mask):
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=int),
            )

        sub_pred = pred[:4, mask]
        cx, cy, bw, bh = sub_pred[0], sub_pred[1], sub_pred[2], sub_pred[3]
        half_w = bw * 0.5
        half_h = bh * 0.5

        x1 = np.clip((cx - half_w - pad_x) * inv_r, 0.0, float(orig_w))
        y1 = np.clip((cy - half_h - pad_y) * inv_r, 0.0, float(orig_h))
        x2 = np.clip((cx + half_w - pad_x) * inv_r, 0.0, float(orig_w))
        y2 = np.clip((cy + half_h - pad_y) * inv_r, 0.0, float(orig_h))

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=int),
            )

        boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=-1).astype(
            np.float32
        )
        scores = max_scores[mask][valid].astype(np.float32)
        classes = best_classes[mask][valid].astype(int)

        raw_dets = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classes,
        )
        return raw_dets.with_nms(threshold=self.nms_threshold)

    def _detect_single_python_onnx(self, frame: np.ndarray) -> sv.Detections:
        """Python fallback for single-frame ONNX letterboxing and inference."""
        tensor, pad_x, pad_y, inv_r, orig_w, orig_h = self._preprocess_letterbox(frame)
        outputs = self.python_session.run(None, {self.input_name: tensor})
        pred = outputs[0][0]  # shape: (8, 8400)
        return self._postprocess_yolo_tensor(pred, orig_w, orig_h, pad_x, pad_y, inv_r)
