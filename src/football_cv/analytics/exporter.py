"""
Structured data serialization and export engine for match analytics records.
Decouples tracking data from video rendering overlays by persisting Player Tracking,
Ball Tracking, Possession Intervals, Candidate Events, and Run Metadata.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from ..possession.events import PossessionInterval
from .event_builder import CandidateEvent


def _json_serialize_default(obj: Any) -> Any:
    """Serialize NumPy primitives and containers into standard JSON types."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class AnalyticsExporter:
    """
    Exports match tracking and event data into standardized CSV and JSON formats.
    """

    def __init__(self, output_dir: str | Path, fps: float = 25.0):
        self.output_dir = Path(output_dir)
        self.fps = fps if fps > 0 else 25.0

    def _ensure_dir(self) -> Path:
        """Create target export directory if it doesn't already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def export_player_tracking(
        self,
        player_tracks: list[dict[int, dict[str, Any]]],
    ) -> tuple[Path, Path]:
        """
        Export frame-by-frame player tracking data to CSV and JSON.

        Returns:
            Tuple of (csv_path, json_path).
        """
        self._ensure_dir()
        csv_path = self.output_dir / "player_tracking.csv"
        json_path = self.output_dir / "player_tracking.json"

        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "player_id",
            "team_id",
            "x_image",
            "y_image",
            "x_pitch",
            "y_pitch",
            "speed_mps",
            "distance_m",
            "has_ball",
            "detection_confidence",
            "position_valid",
        ]

        records: list[dict[str, Any]] = []

        for frame_idx, frame_players in enumerate(player_tracks):
            timestamp = round(frame_idx / self.fps, 3)

            for player_id, p_info in sorted(frame_players.items()):
                img_pos = p_info.get("position_adjusted") or p_info.get("position")
                pitch_pos = p_info.get("position_transformed")

                x_img = round(float(img_pos[0]), 1) if img_pos else None
                y_img = round(float(img_pos[1]), 1) if img_pos else None
                x_pitch = round(float(pitch_pos[0]), 2) if pitch_pos else None
                y_pitch = round(float(pitch_pos[1]), 2) if pitch_pos else None

                speed = (
                    round(float(p_info["speed"]), 2)
                    if p_info.get("speed") is not None
                    else None
                )
                distance = (
                    round(float(p_info["distance_covered"]), 2)
                    if p_info.get("distance_covered") is not None
                    else None
                )

                records.append(
                    {
                        "frame_index": int(frame_idx),
                        "timestamp_seconds": timestamp,
                        "player_id": int(player_id),
                        "team_id": int(p_info["team"])
                        if p_info.get("team") is not None
                        else None,
                        "x_image": x_img,
                        "y_image": y_img,
                        "x_pitch": x_pitch,
                        "y_pitch": y_pitch,
                        "speed_mps": speed,
                        "distance_m": distance,
                        "has_ball": bool(p_info.get("has_ball", False)),
                        "detection_confidence": round(
                            float(p_info.get("confidence", 1.0)), 3
                        ),
                        "position_valid": x_pitch is not None and y_pitch is not None,
                    }
                )

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=_json_serialize_default)

        return csv_path, json_path

    def export_ball_tracking(
        self,
        ball_tracks: list[dict[int, dict[str, Any]]],
        player_tracks: list[dict[int, dict[str, Any]]] | None = None,
    ) -> tuple[Path, Path]:
        """
        Export frame-by-frame ball tracking detections to CSV and JSON.

        Returns:
            Tuple of (csv_path, json_path).
        """
        self._ensure_dir()
        csv_path = self.output_dir / "ball_tracking.csv"
        json_path = self.output_dir / "ball_tracking.json"

        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "x_image",
            "y_image",
            "x_pitch",
            "y_pitch",
            "detection_confidence",
            "is_interpolated",
            "assigned_player_id",
            "assignment_distance",
        ]

        records: list[dict[str, Any]] = []

        for frame_idx, frame_balls in enumerate(ball_tracks):
            timestamp = round(frame_idx / self.fps, 3)
            b_info = frame_balls.get(1, {})

            img_pos = b_info.get("position_adjusted") or b_info.get("position")
            pitch_pos = b_info.get("position_transformed")

            x_img = round(float(img_pos[0]), 1) if img_pos else None
            y_img = round(float(img_pos[1]), 1) if img_pos else None
            x_pitch = round(float(pitch_pos[0]), 2) if pitch_pos else None
            y_pitch = round(float(pitch_pos[1]), 2) if pitch_pos else None

            # Find which player currently holds the ball
            assigned_player_id: int | None = None
            if player_tracks and frame_idx < len(player_tracks):
                for p_id, p_data in player_tracks[frame_idx].items():
                    if p_data.get("has_ball", False):
                        assigned_player_id = p_id
                        break

            records.append(
                {
                    "frame_index": int(frame_idx),
                    "timestamp_seconds": timestamp,
                    "x_image": x_img,
                    "y_image": y_img,
                    "x_pitch": x_pitch,
                    "y_pitch": y_pitch,
                    "detection_confidence": round(
                        float(b_info.get("confidence", 1.0)), 3
                    )
                    if b_info
                    else 0.0,
                    "is_interpolated": bool(b_info.get("is_interpolated", False)),
                    "assigned_player_id": int(assigned_player_id)
                    if assigned_player_id is not None
                    else None,
                    "assignment_distance": round(
                        float(b_info["assignment_distance"]), 2
                    )
                    if "assignment_distance" in b_info
                    else None,
                }
            )

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=_json_serialize_default)

        return csv_path, json_path

    def export_possession_intervals(
        self,
        intervals: list[PossessionInterval],
    ) -> tuple[Path, Path]:
        """
        Export segmented possession intervals to CSV and JSON.

        Returns:
            Tuple of (csv_path, json_path).
        """
        self._ensure_dir()
        csv_path = self.output_dir / "possession_intervals.csv"
        json_path = self.output_dir / "possession_intervals.json"

        fieldnames = [
            "interval_id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "player_id",
            "team_id",
            "duration_seconds",
            "frame_count",
            "confidence",
            "termination_reason",
        ]

        records = [interval.to_dict() for interval in intervals]

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=_json_serialize_default)

        return csv_path, json_path

    def export_events(
        self,
        events: list[CandidateEvent],
    ) -> tuple[Path, Path]:
        """
        Export candidate match events to CSV and JSON.

        Returns:
            Tuple of (csv_path, json_path).
        """
        self._ensure_dir()
        csv_path = self.output_dir / "events.csv"
        json_path = self.output_dir / "events.json"

        fieldnames = [
            "event_id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "from_player_id",
            "to_player_id",
            "from_team_id",
            "to_team_id",
            "start_x_pitch",
            "start_y_pitch",
            "end_x_pitch",
            "end_y_pitch",
            "start_x_image",
            "start_y_image",
            "end_x_image",
            "end_y_image",
            "transition_frames",
            "event_type",
            "confidence",
        ]

        records = [event.to_dict() for event in events]

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=_json_serialize_default)

        return csv_path, json_path

    def export_metadata(
        self,
        config: AppConfig,
        stats: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write run manifest capturing model checkpoints, parameters, and environment state.

        Returns:
            Path to metadata.json.
        """
        self._ensure_dir()
        json_path = self.output_dir / "metadata.json"

        meta = {
            "version": "0.1.0",
            "exported_at": datetime.now(UTC).isoformat(),
            "model": {
                "checkpoint": config.model.path,
                "confidence_threshold": config.model.confidence,
                "device": config.model.device,
                "batch_size": config.model.batch_size,
            },
            "video": {
                "input_path": config.video.input_path,
                "output_path": config.video.output_path,
                "frame_rate": config.video.frame_rate,
                "start_frame": config.video.start_frame,
                "end_frame": config.video.end_frame,
            },
            "parameters": {
                "max_player_ball_distance": config.possession.max_player_ball_distance,
                "minimum_control_frames": config.possession.minimum_control_frames,
                "speed_window_frames": config.movement.speed_window_frames,
                "maximum_transition_frames": config.analytics.pass_network.maximum_transition_frames,
            },
            "summary_statistics": stats or {},
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_serialize_default)

        return json_path

    def export_all(
        self,
        tracks: dict[str, Any],
        intervals: list[PossessionInterval],
        events: list[CandidateEvent],
        config: AppConfig,
    ) -> dict[str, Path]:
        """
        Orchestrate complete export of all tracking and event datasets.

        Returns:
            Dictionary mapping dataset names to generated file paths.
        """
        p_csv, p_json = self.export_player_tracking(tracks.get("players", []))
        b_csv, b_json = self.export_ball_tracking(
            tracks.get("balls", []), tracks.get("players", [])
        )
        i_csv, i_json = self.export_possession_intervals(intervals)
        e_csv, e_json = self.export_events(events)

        stats = {
            "total_frames": len(tracks.get("players", [])),
            "total_possession_intervals": len(intervals),
            "total_events": len(events),
            "event_counts": {},
        }
        for ev in events:
            stats["event_counts"][ev.event_type] = (
                stats["event_counts"].get(ev.event_type, 0) + 1
            )

        m_json = self.export_metadata(config=config, stats=stats)

        return {
            "player_tracking_csv": p_csv,
            "player_tracking_json": p_json,
            "ball_tracking_csv": b_csv,
            "ball_tracking_json": b_json,
            "possession_intervals_csv": i_csv,
            "possession_intervals_json": i_json,
            "events_csv": e_csv,
            "events_json": e_json,
            "metadata_json": m_json,
        }
