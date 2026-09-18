"""
Integration tests for CLI report generation.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from football_cv.cli import main


@pytest.fixture
def dummy_tracking_json(tmp_path: Path) -> Path:
    """Create a dummy player tracking JSON file for report CLI testing."""
    records: list[dict[str, Any]] = []
    for f in range(25):
        records.append(
            {
                "frame_index": f,
                "timestamp_seconds": round(f / 25.0, 2),
                "player_id": 7,
                "team_id": 1,
                "x_pitch": 40.0 + f * 0.5,
                "y_pitch": 25.0 + f * 0.2,
                "has_ball": True,
                "detection_confidence": 0.95,
            }
        )
        records.append(
            {
                "frame_index": f,
                "timestamp_seconds": round(f / 25.0, 2),
                "player_id": 11,
                "team_id": 2,
                "x_pitch": 80.0 - f * 0.3,
                "y_pitch": 45.0 - f * 0.1,
                "has_ball": False,
                "detection_confidence": 0.90,
            }
        )

    json_path = tmp_path / "player_tracking.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f)

    return json_path


class TestCLIReportIntegration:
    """Integration test suite for 'football-cv report' subcommand."""

    def test_cli_report_success(
        self, dummy_tracking_json: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "report_out"
        ret = main(
            [
                "report",
                "--tracks",
                str(dummy_tracking_json),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert ret == 0

        heatmaps_dir = out_dir / "heatmaps"
        data_dir = out_dir / "data"

        assert (heatmaps_dir / "team_1_possession.png").is_file()
        assert (heatmaps_dir / "team_2_possession.png").is_file()
        assert (data_dir / "possession_heatmap.csv").is_file()

    def test_cli_report_with_filters_and_theme(
        self, dummy_tracking_json: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "filtered_report"
        ret = main(
            [
                "report",
                "--tracks",
                str(dummy_tracking_json),
                "--output-dir",
                str(out_dir),
                "--team",
                "1",
                "--theme",
                "classic_turf",
            ]
        )
        assert ret == 0
        assert (out_dir / "heatmaps" / "team_1_possession.png").is_file()

    def test_cli_report_missing_file_fails(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "does_not_exist.json"
        ret = main(["report", "--tracks", str(missing_file)])
        assert ret == 1
