"""
Benchmarking, profiling, and hardware environment instrumentation package for football_cv.
"""

from .environment import EnvironmentCollector
from .exporter import BenchmarkExporter
from .micro import (
    MicroBenchmarkResult,
    benchmark_camera_motion,
    benchmark_geometry,
    benchmark_perspective,
    format_micro_benchmark_table,
    run_all_micro_benchmarks,
    save_micro_benchmark_results,
)
from .profiler import PipelineProfiler, ProfileSummary, StageMetrics, StageTimer
from .runner import BenchmarkReport, BenchmarkRunner, BenchmarkRunResult

__all__ = [
    "BenchmarkExporter",
    "BenchmarkReport",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "EnvironmentCollector",
    "MicroBenchmarkResult",
    "PipelineProfiler",
    "ProfileSummary",
    "StageMetrics",
    "StageTimer",
    "benchmark_camera_motion",
    "benchmark_geometry",
    "benchmark_perspective",
    "format_micro_benchmark_table",
    "run_all_micro_benchmarks",
    "save_micro_benchmark_results",
]
