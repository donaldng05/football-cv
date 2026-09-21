"""
Unit tests for isolated micro-benchmarking module and CLI flag integrations.
"""

import json
from pathlib import Path

import pytest

from football_cv.benchmark.micro import (
    MicroBenchmarkResult,
    benchmark_camera_motion,
    benchmark_geometry,
    benchmark_perspective,
    format_micro_benchmark_table,
    run_all_micro_benchmarks,
    save_micro_benchmark_results,
)
from football_cv.cli import build_parser, handle_benchmark
from football_cv.core import has_cpp_core

pytestmark = [
    pytest.mark.skipif(
        not has_cpp_core(),
        reason="Native C++ extension football_cv._core is not compiled",
    ),
]


class TestMicroBenchmarkKernels:
    """Validate execution and data integrity of micro-benchmarking routines."""

    def test_benchmark_geometry_execution(self):
        """Geometry micro-benchmark should execute and return structured records."""
        results = benchmark_geometry(iterations=500, batch_size=10)
        assert len(results) >= 5
        for res in results:
            assert isinstance(res, MicroBenchmarkResult)
            assert res.category == "Geometry"
            assert res.iterations == 500 or res.iterations == 100
            assert res.python_latency_us > 0.0
            assert res.cpp_latency_us > 0.0
            assert res.speedup > 0.0
            assert res.python_throughput_ops > 0.0
            assert res.cpp_throughput_ops > 0.0

    def test_benchmark_perspective_execution(self):
        """Perspective micro-benchmark should execute and return structured records."""
        results = benchmark_perspective(iterations=200, batch_points=20)
        assert len(results) >= 3
        for res in results:
            assert isinstance(res, MicroBenchmarkResult)
            assert res.category == "Perspective"
            assert res.python_latency_us > 0.0
            assert res.cpp_latency_us > 0.0
            assert res.speedup > 0.0

    def test_benchmark_camera_motion_execution(self):
        """Camera motion feature loop benchmark should execute and return structured records."""
        results = benchmark_camera_motion(iterations=100, num_features=50)
        assert len(results) >= 2
        for res in results:
            assert isinstance(res, MicroBenchmarkResult)
            assert res.category == "Camera Motion"
            assert res.python_latency_us > 0.0
            assert res.cpp_latency_us > 0.0
            assert res.speedup > 0.0

    def test_run_all_micro_benchmarks_scaled(self):
        """Full suite execution with scale=0.005 should return complete summary dict."""
        report = run_all_micro_benchmarks(scale=0.005)
        assert "environment" in report
        assert "summary" in report
        assert "benchmarks" in report

        summary = report["summary"]
        assert summary["total_benchmarks"] >= 10
        assert summary["mean_speedup"] > 0.0
        assert summary["max_speedup"] >= summary["min_speedup"]

    def test_format_micro_benchmark_table(self):
        """Format table should produce aligned ASCII matrix."""
        report = run_all_micro_benchmarks(scale=0.005)
        table_str = format_micro_benchmark_table(report)
        assert "Micro-Benchmark Performance Matrix" in table_str
        assert "Kernel Operation" in table_str
        assert "Speedup" in table_str
        assert "Mean Speedup" in table_str

    def test_save_micro_benchmark_results(self, tmp_path: Path):
        """Saving results should write valid JSON to target directory."""
        report = run_all_micro_benchmarks(scale=0.005)
        out_file = save_micro_benchmark_results(report, tmp_path)
        assert out_file.is_file()
        assert out_file.name == "micro_results.json"

        with open(out_file, encoding="utf-8") as f:
            loaded = json.load(f)
        assert (
            loaded["summary"]["total_benchmarks"]
            == report["summary"]["total_benchmarks"]
        )


class TestCliBenchmarkOptions:
    """Validate CLI argument parsing and dispatching for benchmark subcommand."""

    def test_cli_parser_recognizes_micro_and_backend(self):
        """Parser must parse --backend and --micro flags."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--micro", "--backend", "cpp"])
        assert args.command == "benchmark"
        assert args.micro is True
        assert args.backend == "cpp"

    def test_cli_parser_defaults(self):
        """Default values for benchmark options."""
        parser = build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.micro is False
        assert args.backend is None

    def test_handle_benchmark_micro_dispatch(self, tmp_path: Path, monkeypatch):
        """handle_benchmark with --micro should execute and return exit code 0."""
        parser = build_parser()
        args = parser.parse_args(
            ["benchmark", "--micro", "--output-dir", str(tmp_path)]
        )

        exit_code = handle_benchmark(args)
        assert exit_code == 0
        assert (tmp_path / "micro_results.json").is_file()
