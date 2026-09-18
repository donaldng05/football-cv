"""
Unit tests for possession heatmap generation and tabular export.
"""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from football_cv.analytics.heatmap import (
    HeatmapGenerator,
    PossessionGrid,
    aggregate_possession_grid,
    apply_gaussian_smoothing,
    export_heatmap_csv,
    filter_possession_records,
    render_possession_heatmap,
)
from football_cv.exceptions import AnalyticsError


@pytest.fixture
def sample_possession_records() -> list[dict[str, Any]]:
    """Generate representative possession records spanning Team 1 and Team 2."""
    records: list[dict[str, Any]] = []

    # Team 1, Player 10 possesses ball for 20 frames (0.8s at 25 fps) near left wing
    for f in range(20):
        records.append(
            {
                "frame_index": f,
                "player_id": 10,
                "team_id": 1,
                "has_ball": True,
                "x_pitch": 30.0 + (f * 0.2),
                "y_pitch": 15.0 + (f * 0.1),
                "detection_confidence": 0.95,
            }
        )

    # Team 2, Player 22 possesses ball for 15 frames near center box
    for f in range(20, 35):
        records.append(
            {
                "frame_index": f,
                "player_id": 22,
                "team_id": 2,
                "has_ball": True,
                "x_pitch": 75.0,
                "y_pitch": 34.0,
                "detection_confidence": 0.88,
            }
        )

    # Outfield player without ball (should be filtered out)
    for f in range(10):
        records.append(
            {
                "frame_index": f,
                "player_id": 5,
                "team_id": 1,
                "has_ball": False,
                "x_pitch": 50.0,
                "y_pitch": 30.0,
                "detection_confidence": 0.99,
            }
        )

    # Missing pitch coordinates (should be filtered out)
    records.append(
        {
            "frame_index": 36,
            "player_id": 10,
            "team_id": 1,
            "has_ball": True,
            "x_pitch": None,
            "y_pitch": None,
            "detection_confidence": 0.90,
        }
    )

    # Low confidence record
    records.append(
        {
            "frame_index": 37,
            "player_id": 10,
            "team_id": 1,
            "has_ball": True,
            "x_pitch": 35.0,
            "y_pitch": 20.0,
            "detection_confidence": 0.20,
        }
    )

    return records


class TestPossessionFiltering:
    """Test suite for on-ball record filtering."""

    def test_filter_possession_records_basic(
        self, sample_possession_records: list[dict[str, Any]]
    ) -> None:
        filtered = filter_possession_records(sample_possession_records)
        # 20 frames from Player 10 + 15 frames from Player 22 + 1 low-confidence frame = 36
        assert len(filtered) == 36
        for r in filtered:
            assert bool(r["has_ball"]) is True
            assert r["x_pitch"] is not None
            assert r["y_pitch"] is not None

    def test_filter_by_team(
        self, sample_possession_records: list[dict[str, Any]]
    ) -> None:
        t1 = filter_possession_records(sample_possession_records, team_id=1)
        assert len(t1) == 21  # 20 + 1 low conf
        for r in t1:
            assert r["team_id"] == 1

        t2 = filter_possession_records(sample_possession_records, team_id=2)
        assert len(t2) == 15
        for r in t2:
            assert r["team_id"] == 2

    def test_filter_by_player(
        self, sample_possession_records: list[dict[str, Any]]
    ) -> None:
        p10 = filter_possession_records(sample_possession_records, player_id=10)
        assert len(p10) == 21
        p22 = filter_possession_records(sample_possession_records, player_id=22)
        assert len(p22) == 15
        p99 = filter_possession_records(sample_possession_records, player_id=99)
        assert len(p99) == 0

    def test_filter_by_confidence(
        self, sample_possession_records: list[dict[str, Any]]
    ) -> None:
        filtered = filter_possession_records(
            sample_possession_records, min_confidence=0.50
        )
        assert len(filtered) == 35  # The record with conf 0.20 is excluded

    def test_filter_by_frame_range(
        self, sample_possession_records: list[dict[str, Any]]
    ) -> None:
        filtered = filter_possession_records(
            sample_possession_records, start_frame=5, end_frame=15
        )
        assert len(filtered) == 11
        for r in filtered:
            assert 5 <= r["frame_index"] <= 15


class TestGridAggregation:
    """Test suite for 2D duration binning and Gaussian smoothing."""

    def test_aggregate_empty_records(self) -> None:
        grid = aggregate_possession_grid([], grid_width=30, grid_height=20)
        assert isinstance(grid, PossessionGrid)
        assert grid.raw_grid.shape == (20, 30)
        assert grid.smoothed_grid.shape == (20, 30)
        assert grid.total_possession_seconds == 0.0
        assert grid.total_frames == 0

    def test_aggregate_duration_weighting(
        self, sample_possession_records: list[dict[str, Any]]
    ) -> None:
        # Filter for Team 1: 20 frames at 25 fps = 0.8s (ignoring low conf)
        records = filter_possession_records(
            sample_possession_records, team_id=1, min_confidence=0.5
        )
        grid = aggregate_possession_grid(
            records, grid_width=60, grid_height=40, fps=25.0
        )

        assert grid.total_frames == 20
        assert pytest.approx(grid.total_possession_seconds, 0.01) == 0.8
        assert pytest.approx(np.sum(grid.raw_grid), 0.01) == 0.8
        assert np.all(grid.smoothed_grid >= 0.0)

    def test_aggregate_invalid_dimensions(self) -> None:
        with pytest.raises(AnalyticsError, match="must be >= 2"):
            aggregate_possession_grid([], grid_width=1, grid_height=40)

        with pytest.raises(AnalyticsError, match="must be > 0"):
            aggregate_possession_grid([], pitch_length=-10.0)

    def test_apply_gaussian_smoothing(self) -> None:
        raw = np.zeros((10, 10))
        raw[5, 5] = 1.0

        # Zero sigma returns copy
        assert np.array_equal(apply_gaussian_smoothing(raw, sigma=0.0), raw)

        # Positive sigma disperses peak
        smoothed = apply_gaussian_smoothing(raw, sigma=1.0)
        assert smoothed[5, 5] < 1.0
        assert smoothed[5, 6] > 0.0
        assert np.all(smoothed >= 0.0)


class TestHeatmapRenderingAndExport:
    """Test suite for rendering PNG figures and exporting CSV data."""

    def test_render_possession_heatmap(
        self, sample_possession_records: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        records = filter_possession_records(sample_possession_records)
        grid = aggregate_possession_grid(records)
        out_png = tmp_path / "heatmap_test.png"

        fig, _ = render_possession_heatmap(
            grid=grid,
            output_path=out_png,
            title="Match Possession Test",
            theme="tactical_dark",
        )
        assert out_png.is_file()
        assert out_png.stat().st_size > 1000  # Non-trivial image file

        matplotlib.pyplot.close(fig)

    def test_export_heatmap_csv(
        self, sample_possession_records: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        records = filter_possession_records(sample_possession_records, team_id=1)
        grid = aggregate_possession_grid(records, grid_width=20, grid_height=10)
        out_csv = tmp_path / "possession_heatmap.csv"

        export_heatmap_csv(grid, out_csv, team_id=1)
        assert out_csv.is_file()

        with open(out_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0
            first = rows[0]
            assert "team_id" in first
            assert "grid_x_bin" in first
            assert "grid_y_bin" in first
            assert "possession_seconds" in first
            assert "density_share_percent" in first
            total_sec = sum(float(r["possession_seconds"]) for r in rows)
            assert pytest.approx(total_sec, 0.01) == grid.total_possession_seconds
            assert any(float(r["smoothed_density"]) > 0.0 for r in rows)


class TestHeatmapGenerator:
    """Test suite for the high-level HeatmapGenerator coordinator."""

    def test_generate_from_records_and_files(
        self, sample_possession_records: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "report"
        generator = HeatmapGenerator(output_dir=output_dir)

        # 1. Test generate_from_records
        artifacts = generator.generate_from_records(sample_possession_records)
        assert "team_1" in artifacts
        assert "team_2" in artifacts
        assert "heatmap_csv" in artifacts
        assert artifacts["team_1"].is_file()
        assert artifacts["team_2"].is_file()
        assert artifacts["heatmap_csv"].is_file()

        # 2. Test generate_from_file with JSON
        json_file = tmp_path / "player_tracking.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_possession_records, f)

        artifacts_json = generator.generate_from_file(json_file)
        assert len(artifacts_json) >= 3

        # 3. Test generate_from_file with CSV
        csv_file = tmp_path / "player_tracking.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(sample_possession_records[0].keys())
            )
            writer.writeheader()
            writer.writerows(sample_possession_records)

        artifacts_csv = generator.generate_from_file(csv_file)
        assert len(artifacts_csv) >= 3

    def test_generate_player_filter(
        self, sample_possession_records: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "report_player"
        generator = HeatmapGenerator(output_dir=output_dir)
        artifacts = generator.generate_from_records(
            sample_possession_records, player_filter=10
        )
        assert "player_10" in artifacts
        assert artifacts["player_10"].is_file()

    def test_load_records_invalid_file(self, tmp_path: Path) -> None:
        generator = HeatmapGenerator(output_dir=tmp_path)
        with pytest.raises(AnalyticsError, match="not found"):
            generator.load_records_from_file(tmp_path / "nonexistent.json")

        bad_file = tmp_path / "unsupported.txt"
        bad_file.write_text("dummy", encoding="utf-8")
        with pytest.raises(AnalyticsError, match="Unsupported file format"):
            generator.load_records_from_file(bad_file)
