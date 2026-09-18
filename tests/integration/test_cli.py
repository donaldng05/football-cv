"""
Integration tests for CLI entrypoint and commands.
"""

import pytest

from football_cv.cli import main


class TestCLIIntegration:
    def test_cli_version(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "football-cv" in captured.out or "football-cv" in captured.err

    def test_cli_validate_success(self):
        ret = main(["validate", "--config", "configs/default.yaml"])
        assert ret == 0

    def test_cli_validate_missing_file_returns_error(self):
        ret = main(["validate", "--config", "configs/does_not_exist.yaml"])
        assert ret == 1

    def test_cli_analyze_help(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["analyze", "--help"])
        assert excinfo.value.code == 0

    def test_cli_report_help(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["report", "--help"])
        assert excinfo.value.code == 0

    def test_cli_benchmark_help(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["benchmark", "--help"])
        assert excinfo.value.code == 0

    def test_cli_analyze_execution_smoke(self, tmp_path):
        out_vid = tmp_path / "cli_out.avi"
        stub_file = tmp_path / "cli_track_stubs.pkl"
        ret = main(
            [
                "analyze",
                "--config",
                "configs/fast.yaml",
                "--start-frame",
                "0",
                "--end-frame",
                "2",
                "--output",
                str(out_vid),
                "--stub-path",
                str(stub_file),
            ]
        )
        assert ret == 0
        assert out_vid.is_file()
        assert out_vid.stat().st_size > 0
