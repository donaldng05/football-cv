"""
Reporting and data export utility for pipeline benchmarking runs.
"""

import csv
import json
import logging
from pathlib import Path

from .runner import BenchmarkReport

logger = logging.getLogger(__name__)


class BenchmarkExporter:
    """
    Exports benchmark runs into structured CSV, JSON, and terminal-friendly report cards.
    """

    @classmethod
    def export_csv(cls, report: BenchmarkReport, output_path: str | Path) -> Path:
        """
        Export benchmark runs to a flat tabular CSV file.

        Args:
            report: Populated BenchmarkReport.
            output_path: Target CSV file path.

        Returns:
            Path to written CSV file.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "benchmark_id",
            "run_id",
            "run_name",
            "created_at",
            "model_path",
            "engine",
            "confidence",
            "batch_size",
            "device",
            "num_frames",
            "total_duration_seconds",
            "throughput_fps",
            "video_decoding_ms",
            "detection_and_tracking_ms",
            "camera_motion_ms",
            "perspective_transform_ms",
            "ball_interpolation_ms",
            "speed_distance_ms",
            "team_assignment_ms",
            "possession_assignment_ms",
            "analytics_inference_ms",
            "rendering_ms",
        ]

        rows = []
        for run in report.runs:
            prof = run.profile
            row = {
                "benchmark_id": report.benchmark_id,
                "run_id": run.run_id,
                "run_name": run.run_name,
                "created_at": report.created_at,
                "model_path": run.model_path,
                "engine": getattr(run, "engine", "ultralytics"),
                "confidence": run.confidence,
                "batch_size": run.batch_size,
                "device": run.device,
                "num_frames": run.num_frames,
                "total_duration_seconds": prof.total_duration_seconds,
                "throughput_fps": prof.throughput_fps,
                "video_decoding_ms": prof.stages.get("video_decoding", {}).ms_per_frame
                if "video_decoding" in prof.stages
                else 0.0,
                "detection_and_tracking_ms": prof.stages.get(
                    "detection_and_tracking", {}
                ).ms_per_frame
                if "detection_and_tracking" in prof.stages
                else 0.0,
                "camera_motion_ms": prof.stages.get("camera_motion", {}).ms_per_frame
                if "camera_motion" in prof.stages
                else 0.0,
                "perspective_transform_ms": prof.stages.get(
                    "perspective_transform", {}
                ).ms_per_frame
                if "perspective_transform" in prof.stages
                else 0.0,
                "ball_interpolation_ms": prof.stages.get(
                    "ball_interpolation", {}
                ).ms_per_frame
                if "ball_interpolation" in prof.stages
                else 0.0,
                "speed_distance_ms": prof.stages.get("speed_distance", {}).ms_per_frame
                if "speed_distance" in prof.stages
                else 0.0,
                "team_assignment_ms": prof.stages.get(
                    "team_assignment", {}
                ).ms_per_frame
                if "team_assignment" in prof.stages
                else 0.0,
                "possession_assignment_ms": prof.stages.get(
                    "possession_assignment", {}
                ).ms_per_frame
                if "possession_assignment" in prof.stages
                else 0.0,
                "analytics_inference_ms": prof.stages.get(
                    "analytics_inference", {}
                ).ms_per_frame
                if "analytics_inference" in prof.stages
                else 0.0,
                "rendering_ms": prof.stages.get("rendering", {}).ms_per_frame
                if "rendering" in prof.stages
                else 0.0,
            }
            rows.append(row)

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Saved benchmark CSV results to {target}")
        return target

    @classmethod
    def export_json(cls, report: BenchmarkReport, output_path: str | Path) -> Path:
        """Export comprehensive benchmark report to JSON."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved benchmark JSON report to {target}")
        return target

    @classmethod
    def format_terminal_table(cls, report: BenchmarkReport) -> str:
        """Format an ASCII report summary card for terminal output."""
        lines = [
            "=" * 78,
            f"  football-cv Performance Benchmark Report: {report.benchmark_id}",
            "=" * 78,
            f"Timestamp: {report.created_at}",
            f"Platform:  {report.environment.get('system', {}).get('platform', 'unknown')}",
            f"CPU:       {report.environment.get('cpu', {}).get('model', 'unknown')} ({report.environment.get('cpu', {}).get('logical_cores', '?')} threads)",
            f"RAM:       {report.environment.get('memory', {}).get('total_gb', '?')} GB",
            f"PyTorch:   {report.environment.get('packages', {}).get('torch', 'unknown')} (CUDA: {report.environment.get('accelerator', {}).get('cuda_available', False)})",
            "-" * 78,
            f"{'Run Name':<28} {'Frames':<8} {'Total (s)':<11} {'FPS':<9} {'Top Bottleneck':<20}",
            "-" * 78,
        ]

        for r in report.runs:
            prof = r.profile
            # Identify top latency contributor stage
            top_stage = "None"
            top_pct = 0.0
            for s_name, s_met in prof.stages.items():
                if s_met.percent_of_total > top_pct:
                    top_pct = s_met.percent_of_total
                    top_stage = f"{s_name} ({top_pct:.1f}%)"

            lines.append(
                f"{r.run_name:<28} {r.num_frames:<8} {prof.total_duration_seconds:<11.2f} {prof.throughput_fps:<9.1f} {top_stage:<20}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)
