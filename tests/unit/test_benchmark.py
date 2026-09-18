"""
Unit tests for the benchmarking and profiling subsystem.
"""

import csv
import json
import time
from pathlib import Path

import pytest

from football_cv.benchmark.environment import EnvironmentCollector
from football_cv.benchmark.exporter import BenchmarkExporter
from football_cv.benchmark.profiler import PipelineProfiler, ProfileSummary
from football_cv.benchmark.runner import (
    BenchmarkReport,
    BenchmarkRunner,
    BenchmarkRunResult,
)
from football_cv.config import load_config
from football_cv.exceptions import BenchmarkError


class TestEnvironmentCollector:
    """Test suite for EnvironmentCollector."""

    def test_collect_returns_system_manifest(self) -> None:
        manifest = EnvironmentCollector.collect()

        assert "timestamp_utc" in manifest
        assert "system" in manifest
        assert "os" in manifest["system"]
        assert "cpu" in manifest
        assert "logical_cores" in manifest["cpu"]
        assert "packages" in manifest
        assert manifest["packages"]["python"] is not None

    def test_export_json(self, tmp_path: Path) -> None:
        out_json = tmp_path / "environment.json"
        path = EnvironmentCollector.export_json(out_json)

        assert path.is_file()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            assert "system" in data
            assert "cpu" in data


class TestPipelineProfiler:
    """Test suite for StageTimer and PipelineProfiler."""

    def test_stage_timer_records_latency(self) -> None:
        profiler = PipelineProfiler()

        with profiler.time_stage("test_stage"):
            time.sleep(0.01)

        assert "test_stage" in profiler.stage_timings
        assert profiler.stage_timings["test_stage"] >= 0.008

    def test_summarize_metrics(self) -> None:
        profiler = PipelineProfiler()
        profiler.record_stage("stage_a", 0.04)
        profiler.record_stage("stage_b", 0.06)

        summary = profiler.summarize(num_frames=10)

        assert isinstance(summary, ProfileSummary)
        assert summary.num_frames == 10
        assert pytest.approx(summary.total_duration_seconds, 0.001) == 0.10
        assert summary.throughput_fps == 100.0  # 10 frames / 0.10s = 100 fps

        # Stage A should be 40%
        assert summary.stages["stage_a"].percent_of_total == 40.0
        assert summary.stages["stage_a"].ms_per_frame == 4.0

        # Stage B should be 60%
        assert summary.stages["stage_b"].percent_of_total == 60.0
        assert summary.stages["stage_b"].ms_per_frame == 6.0

    def test_profiler_reset(self) -> None:
        profiler = PipelineProfiler()
        profiler.record_stage("stage_a", 0.05)
        assert len(profiler.stage_timings) == 1
        profiler.reset()
        assert len(profiler.stage_timings) == 0


class TestBenchmarkExporter:
    """Test suite for BenchmarkExporter."""

    @pytest.fixture
    def sample_report(self) -> BenchmarkReport:
        profiler = PipelineProfiler()
        profiler.record_stage("video_decoding", 0.02)
        profiler.record_stage("detection_and_tracking", 0.08)
        summary = profiler.summarize(num_frames=20)

        run = BenchmarkRunResult(
            run_id="run_1234",
            run_name="fast_run",
            model_path="models/best.pt",
            confidence=0.2,
            batch_size=10,
            device="cpu",
            num_frames=20,
            profile=summary,
        )

        return BenchmarkReport(
            benchmark_id="bench_test_01",
            created_at="2026-09-17T22:00:00Z",
            environment=EnvironmentCollector.collect(),
            runs=[run],
        )

    def test_export_csv(self, sample_report: BenchmarkReport, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        path = BenchmarkExporter.export_csv(sample_report, csv_path)

        assert path.is_file()
        with open(path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            row = reader[0]
            assert row["benchmark_id"] == "bench_test_01"
            assert row["run_name"] == "fast_run"
            assert float(row["throughput_fps"]) > 0.0
            assert "video_decoding_ms" in row
            assert "detection_and_tracking_ms" in row

    def test_export_json(self, sample_report: BenchmarkReport, tmp_path: Path) -> None:
        json_path = tmp_path / "results.json"
        path = BenchmarkExporter.export_json(sample_report, json_path)

        assert path.is_file()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            assert data["benchmark_id"] == "bench_test_01"
            assert len(data["runs"]) == 1

    def test_format_terminal_table(self, sample_report: BenchmarkReport) -> None:
        table = BenchmarkExporter.format_terminal_table(sample_report)
        assert "football-cv Performance Benchmark Report" in table
        assert "fast_run" in table
        assert "FPS" in table


class TestBenchmarkRunner:
    """Test suite for BenchmarkRunner."""

    def test_missing_video_raises_benchmark_error(self, tmp_path: Path) -> None:
        config = load_config("configs/fast.yaml")
        config.video.input_path = str(tmp_path / "nonexistent.mp4")

        runner = BenchmarkRunner()
        with pytest.raises(BenchmarkError, match="not found"):
            runner.run_single(config, num_frames=10)

    def test_run_single_execution(self) -> None:
        config = load_config("configs/fast.yaml")
        runner = BenchmarkRunner(warmup_frames=2)

        result = runner.run_single(config, num_frames=5, run_name="unit_test_run")
        assert isinstance(result, BenchmarkRunResult)
        assert result.num_frames == 5
        assert result.profile.total_duration_seconds > 0.0
        assert result.profile.throughput_fps > 0.0
        assert len(result.profile.stages) >= 5

    def test_run_sweep_execution(self) -> None:
        config = load_config("configs/fast.yaml")
        runner = BenchmarkRunner(warmup_frames=0)

        report = runner.run_sweep(
            config=config,
            num_frames=5,
            confidences=[0.20, 0.30],
        )

        assert isinstance(report, BenchmarkReport)
        assert len(report.runs) == 2
        assert report.runs[0].confidence == 0.20
        assert report.runs[1].confidence == 0.30
