"""
High-resolution stage profiler and latency timer for pipeline benchmarking.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StageMetrics:
    """Performance metrics for a single pipeline processing stage."""

    stage_name: str
    total_seconds: float
    ms_per_frame: float
    percent_of_total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileSummary:
    """Aggregated latency and throughput summary for a benchmark run."""

    num_frames: int
    total_duration_seconds: float
    throughput_fps: float
    stages: dict[str, StageMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "total_duration_seconds": round(self.total_duration_seconds, 4),
            "throughput_fps": round(self.throughput_fps, 2),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }


class StageTimer:
    """Context manager for high-resolution timing of an individual stage."""

    def __init__(self, stage_name: str, profiler: "PipelineProfiler"):
        self.stage_name = stage_name
        self.profiler = profiler
        self.start_time: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.profiler.record_stage(self.stage_name, elapsed)


class PipelineProfiler:
    """
    Tracks and aggregates stage-by-stage wall-clock latency throughout execution.
    """

    def __init__(self) -> None:
        self.stage_timings: dict[str, float] = {}

    def time_stage(self, stage_name: str) -> StageTimer:
        """Return a context manager to measure wall-clock duration of a stage."""
        return StageTimer(stage_name, self)

    def record_stage(self, stage_name: str, duration_seconds: float) -> None:
        """Record elapsed duration for a specific stage."""
        self.stage_timings[stage_name] = self.stage_timings.get(stage_name, 0.0) + max(
            0.0, duration_seconds
        )

    def reset(self) -> None:
        """Clear all recorded stage timings."""
        self.stage_timings.clear()

    def summarize(self, num_frames: int) -> ProfileSummary:
        """
        Compute summary metrics, ms/frame, percentage share, and throughput FPS.

        Args:
            num_frames: Number of video frames processed during profiling.

        Returns:
            ProfileSummary containing stage breakdowns and overall FPS.
        """
        safe_frames = max(1, num_frames)
        total_time = sum(self.stage_timings.values())
        safe_total = total_time if total_time > 0 else 1e-6

        fps = round(safe_frames / safe_total, 2)

        stages: dict[str, StageMetrics] = {}
        for stage_name, duration in self.stage_timings.items():
            ms_frame = (duration * 1000.0) / safe_frames
            pct = (duration / safe_total) * 100.0
            stages[stage_name] = StageMetrics(
                stage_name=stage_name,
                total_seconds=round(duration, 4),
                ms_per_frame=round(ms_frame, 2),
                percent_of_total=round(pct, 2),
            )

        return ProfileSummary(
            num_frames=num_frames,
            total_duration_seconds=round(total_time, 4),
            throughput_fps=fps,
            stages=stages,
        )
