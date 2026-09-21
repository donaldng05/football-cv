"""
Multi-object tracking engine using ByteTrack and supervision.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from ..utils.geometry import get_center_of_bbox, get_foot_position
from .detector import ObjectDetector


class ObjectTracker:
    """Handles object detection, tracking, and position attribution."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.10,
        batch_size: int = 20,
        device: str | None = None,
        engine: str = "ultralytics",
        nms_threshold: float = 0.50,
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence=confidence,
            batch_size=batch_size,
            device=device,
            engine=engine,
            nms_threshold=nms_threshold,
        )
        self.tracker = sv.ByteTrack()

    def add_positions_to_tracks(self, tracks: dict[str, Any]) -> None:
        """Calculate and add center (ball) or foot position (players/referees) to tracks."""
        for obj_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if obj_name == "ball" or obj_name == "balls":
                        pos = get_center_of_bbox(bbox)
                    else:
                        pos = get_foot_position(bbox)
                    tracks[obj_name][frame_num][track_id]["position"] = pos

    def get_tracked_objects(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Detect and track objects across frames, with optional pickle stub caching.

        Args:
            frames: List of video frames.
            read_from_stub: If True and stub_path exists, load tracks from stub.
            stub_path: Filepath for stub cache.

        Returns:
            Dictionary with 'players', 'referees', 'balls' frame tracks.
        """
        if read_from_stub and stub_path is not None and Path(stub_path).is_file():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detector.detect_frames(frames)

        tracks: dict[str, list[dict[int, Any]]] = {
            "players": [],
            "referees": [],
            "balls": [],
        }

        cls_names = self.detector.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        player_cls_id = cls_names_inv.get("player", 2)
        goalkeeper_cls_id = cls_names_inv.get("goalkeeper", 1)
        referee_cls_id = cls_names_inv.get("referee", 3)
        ball_cls_id = cls_names_inv.get("ball", 0)

        for frame_num, detection in enumerate(detections):
            # Support both sv.Detections (ONNX) and Ultralytics Results
            if isinstance(detection, sv.Detections):
                detection_supervision = detection
            else:
                detection_supervision = sv.Detections.from_ultralytics(detection)

            # Map goalkeeper class to player class so ByteTrack tracks them consistently
            if (
                detection_supervision.class_id is not None
                and len(detection_supervision.class_id) > 0
            ):
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if class_id == goalkeeper_cls_id:
                        detection_supervision.class_id[object_ind] = player_cls_id

            # Update ByteTrack tracker
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["balls"].append({})

            for frame_det in detection_with_tracks:
                bbox = frame_det[0].tolist()
                cls_id = frame_det[3]
                track_id = int(frame_det[4])

                if cls_id == player_cls_id:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == referee_cls_id:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Extract ball detections directly from detector (ball is not passed through ByteTrack)
            for frame_det in detection_supervision:
                bbox = frame_det[0].tolist()
                cls_id = frame_det[3]

                if cls_id == ball_cls_id:
                    tracks["balls"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            stub_p = Path(stub_path)
            stub_p.parent.mkdir(parents=True, exist_ok=True)
            with open(stub_p, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_annotations(
        self,
        frames: list[np.ndarray],
        tracks: dict[str, Any],
        team_ball_control: np.ndarray,
    ) -> list[np.ndarray]:
        """Backward-compatible wrapper delegating to FrameAnnotator."""
        from ..rendering.annotations import FrameAnnotator

        return FrameAnnotator().draw_annotations(frames, tracks, team_ball_control)


# Alias for backward compatibility
Tracker = ObjectTracker
