"""
Integration tests for the 'football-cv error-analysis' CLI subcommand.
"""

import csv
import json
from pathlib import Path

import pytest

from football_cv.cli import main


class TestCLIErrorAnalysisIntegration:
    """Integration test suite for error-analysis CLI subcommand."""

    def test_cli_error_analysis_help(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["error-analysis", "--help"])
        assert excinfo.value.code == 0

    def test_cli_error_analysis_run(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "error_reports"
        ret = main(
            [
                "error-analysis",
                "--config",
                "configs/default.yaml",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert ret == 0
        json_path = out_dir / "error_report.json"
        csv_path = out_dir / "mitigation_summary.csv"

        assert json_path.is_file()
        assert csv_path.is_file()

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_cases"] == 5
            assert data["passed_cases"] == 5
            assert data["pass_rate_percent"] == 100.0

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 5
            case_ids = {r["case_id"] for r in rows}
            assert "TS_PASS_01" in case_ids
            assert "SH_VEL_02" in case_ids
            assert "POSS_FLK_03" in case_ids
            assert "CAM_CUT_04" in case_ids
            assert "BALL_DROP_05" in case_ids

    def test_cli_error_analysis_invalid_config(self, tmp_path: Path) -> None:
        ret = main(
            [
                "error-analysis",
                "--config",
                "configs/non_existent.yaml",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert ret == 1
