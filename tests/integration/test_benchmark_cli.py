"""
Integration tests for the 'football-cv benchmark' CLI subcommand.
"""

import csv
from pathlib import Path

from football_cv.cli import main


class TestCLIBenchmarkIntegration:
    """Integration test suite for benchmark CLI subcommand."""

    def test_cli_benchmark_default_run(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "benchmarks"
        ret = main(
            [
                "benchmark",
                "--config",
                "configs/fast.yaml",
                "--num-frames",
                "5",
                "--warmup",
                "1",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert ret == 0
        assert (out_dir / "results.csv").is_file()
        assert (out_dir / "results.json").is_file()
        assert (out_dir / "environment.json").is_file()

        with open(out_dir / "results.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert float(rows[0]["throughput_fps"]) > 0.0

    def test_cli_benchmark_confidence_sweep(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "sweep_benchmarks"
        ret = main(
            [
                "benchmark",
                "--config",
                "configs/fast.yaml",
                "--num-frames",
                "5",
                "--warmup",
                "0",
                "--confidences",
                "0.20,0.30",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert ret == 0
        with open(out_dir / "results.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2

    def test_cli_benchmark_missing_config_returns_error(self, tmp_path: Path) -> None:
        ret = main(
            [
                "benchmark",
                "--config",
                str(tmp_path / "missing.yaml"),
            ]
        )
        assert ret == 1
