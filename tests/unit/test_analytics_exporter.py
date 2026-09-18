"""
Unit tests for structured analytics serialization and export engine.
"""

import csv
import json

from football_cv.analytics.event_builder import CandidateEvent
from football_cv.analytics.exporter import AnalyticsExporter
from football_cv.config import load_config
from football_cv.possession.events import PossessionInterval


class TestAnalyticsExporter:
    def test_export_player_tracking(self, tmp_path):
        exporter = AnalyticsExporter(output_dir=tmp_path / "exports", fps=25.0)
        player_tracks = [
            {
                1: {
                    "team": 1,
                    "position": (100, 200),
                    "position_transformed": [10.5, 20.2],
                    "speed": 3.45,
                    "distance_covered": 12.5,
                    "has_ball": True,
                    "confidence": 0.95,
                }
            }
        ]
        csv_path, json_path = exporter.export_player_tracking(player_tracks)

        assert csv_path.is_file()
        assert json_path.is_file()

        # Verify CSV
        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            row = reader[0]
            assert int(row["frame_index"]) == 0
            assert int(row["player_id"]) == 1
            assert int(row["team_id"]) == 1
            assert float(row["speed_mps"]) == 3.45
            assert row["has_ball"] == "True"
            assert row["position_valid"] == "True"

        # Verify JSON
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["player_id"] == 1
            assert data[0]["x_pitch"] == 10.5

    def test_export_ball_tracking(self, tmp_path):
        exporter = AnalyticsExporter(output_dir=tmp_path / "exports", fps=25.0)
        ball_tracks = [
            {
                1: {
                    "position": (150, 250),
                    "position_transformed": [12.0, 22.0],
                    "confidence": 0.88,
                    "is_interpolated": True,
                    "assignment_distance": 25.4,
                }
            }
        ]
        player_tracks = [{7: {"has_ball": True}}]
        csv_path, json_path = exporter.export_ball_tracking(ball_tracks, player_tracks)

        assert csv_path.is_file()
        assert json_path.is_file()

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["is_interpolated"] is True
            assert data[0]["assigned_player_id"] == 7
            assert data[0]["assignment_distance"] == 25.4

    def test_export_possession_intervals(self, tmp_path):
        exporter = AnalyticsExporter(output_dir=tmp_path / "exports", fps=25.0)
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=10,
                start_time=0.0,
                end_time=0.4,
                player_id=9,
                team_id=1,
                duration_seconds=0.44,
                frame_count=11,
                termination_reason="pass",
            )
        ]
        csv_path, json_path = exporter.export_possession_intervals(intervals)

        assert csv_path.is_file()
        assert json_path.is_file()

        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert int(reader[0]["player_id"]) == 9
            assert reader[0]["termination_reason"] == "pass"

    def test_export_events(self, tmp_path):
        exporter = AnalyticsExporter(output_dir=tmp_path / "exports", fps=25.0)
        events = [
            CandidateEvent(
                event_id=1,
                start_frame=5,
                end_frame=8,
                start_time=0.2,
                end_time=0.32,
                from_player_id=9,
                to_player_id=11,
                from_team_id=1,
                to_team_id=1,
                start_x_pitch=20.0,
                start_y_pitch=10.0,
                end_x_pitch=35.0,
                end_y_pitch=15.0,
                start_x_image=200.0,
                start_y_image=100.0,
                end_x_image=350.0,
                end_y_image=150.0,
                transition_frames=3,
                event_type="candidate_pass",
                confidence=0.85,
            )
        ]
        csv_path, json_path = exporter.export_events(events)

        assert csv_path.is_file()
        assert json_path.is_file()

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["event_type"] == "candidate_pass"
            assert data[0]["from_player_id"] == 9
            assert data[0]["to_player_id"] == 11

    def test_export_metadata(self, tmp_path):
        exporter = AnalyticsExporter(output_dir=tmp_path / "exports", fps=25.0)
        config = load_config("configs/fast.yaml")
        meta_path = exporter.export_metadata(config=config, stats={"custom_metric": 42})

        assert meta_path.is_file()
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            assert meta["version"] == "0.1.0"
            assert "model" in meta
            assert "video" in meta
            assert meta["summary_statistics"]["custom_metric"] == 42

    def test_export_all_orchestrator(self, tmp_path):
        exporter = AnalyticsExporter(output_dir=tmp_path / "all_exports", fps=25.0)
        config = load_config("configs/fast.yaml")

        tracks = {
            "players": [{1: {"has_ball": True, "team": 1}}],
            "balls": [{1: {"confidence": 0.9}}],
            "referees": [],
        }
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=0,
                start_time=0.0,
                end_time=0.0,
                player_id=1,
                team_id=1,
                duration_seconds=0.04,
                frame_count=1,
            )
        ]
        events = [
            CandidateEvent(
                event_id=1,
                start_frame=0,
                end_frame=0,
                start_time=0.0,
                end_time=0.0,
                from_player_id=1,
                to_player_id=1,
                from_team_id=1,
                to_team_id=1,
                start_x_pitch=None,
                start_y_pitch=None,
                end_x_pitch=None,
                end_y_pitch=None,
                start_x_image=None,
                start_y_image=None,
                end_x_image=None,
                end_y_image=None,
                transition_frames=0,
                event_type="recovery",
            )
        ]

        paths = exporter.export_all(tracks, intervals, events, config)
        assert len(paths) == 9
        for _key, p in paths.items():
            assert p.is_file()
            assert p.stat().st_size > 0
