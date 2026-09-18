"""
Rigorous error analysis subsystem for computer vision football tracking pipelines.

Provides failure taxonomy, diagnostic edge cases, before/after mitigation
evaluators, and structured report exporters.
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..camera_motion.estimator import CameraMotionEstimator
from ..possession.events import PossessionInterval, PossessionIntervalExtractor
from .event_builder import EventBuilder, EventType


class FailureCategory(StrEnum):
    """Broad pipeline stage classification of failure modes."""

    DETECTION = "detection"
    TRACKING = "tracking"
    PERSPECTIVE_MOTION = "perspective_motion"
    ANALYTICS = "analytics"


class FailureMode(StrEnum):
    """Specific behavioral failure patterns observed in computer vision pipelines."""

    BALL_DROPOUT = "ball_dropout"
    POSSESSION_FLICKER = "possession_flicker"
    TRACK_SWAP_PASS = "track_swap_pass"
    CAMERA_CUT_DISRUPTION = "camera_cut_disruption"
    SUPERHUMAN_VELOCITY = "superhuman_velocity"


@dataclass(frozen=True)
class FailureCase:
    """Specification of a reproducible failure benchmark case."""

    case_id: str
    name: str
    category: FailureCategory
    mode: FailureMode
    description: str
    root_cause: str
    mitigation_strategy: str
    metric_name: str
    unit: str = "count"


@dataclass
class CaseEvaluationResult:
    """Outcome of running a diagnostic failure case before and after mitigation."""

    case_id: str
    name: str
    category: str
    mode: str
    metric_name: str
    unit: str
    baseline_metric: float
    mitigated_metric: float
    delta: float
    improvement_percent: float
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to a serializable dictionary."""
        return asdict(self)


@dataclass
class ErrorReport:
    """Summary report aggregating diagnostic failure case outcomes."""

    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate_percent: float
    results: list[CaseEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a serializable dictionary."""
        return asdict(self)


# Standard failure cases curated for Phase 10
BENCHMARK_CASES: list[FailureCase] = [
    FailureCase(
        case_id="TS_PASS_01",
        name="Track-Swap False Pass Rejection",
        category=FailureCategory.TRACKING,
        mode=FailureMode.TRACK_SWAP_PASS,
        description="Tracker identity switch on the same physical player creating false passes.",
        root_cause="Occlusion or model confidence drop assigning new track ID with near-zero displacement.",
        mitigation_strategy="Kinematic displacement check rejecting instantaneous transfers (< 1.5m, <= 2 frames).",
        metric_name="false_pass_count",
        unit="count",
    ),
    FailureCase(
        case_id="SH_VEL_02",
        name="Superhuman Velocity Jump Filtering",
        category=FailureCategory.TRACKING,
        mode=FailureMode.SUPERHUMAN_VELOCITY,
        description="Player track teleportation across the pitch registering impossible passes.",
        root_cause="False positive detection or tracker bounding box jump across distant players.",
        mitigation_strategy="Maximum pass velocity threshold filter (> 45.0 m/s / 162 km/h).",
        metric_name="superhuman_pass_count",
        unit="count",
    ),
    FailureCase(
        case_id="POSS_FLK_03",
        name="Scrum Possession Flicker Suppression",
        category=FailureCategory.ANALYTICS,
        mode=FailureMode.POSSESSION_FLICKER,
        description="Equidistant scrum contest causing 1-frame oscillation between opposing players.",
        root_cause="Close proximity Euclidean tie-breaks fluctuating frame-by-frame.",
        mitigation_strategy="Temporal hysteresis buffer requiring >= 2 frames to confirm turnover.",
        metric_name="false_turnover_count",
        unit="count",
    ),
    FailureCase(
        case_id="CAM_CUT_04",
        name="Camera Cut Discontinuity Reset",
        category=FailureCategory.PERSPECTIVE_MOTION,
        mode=FailureMode.CAMERA_CUT_DISRUPTION,
        description="Broadcast scene cuts inducing optical flow drift and homography explosion.",
        root_cause="Sparse optical flow tracking across discontinuous scene transitions.",
        mitigation_strategy="Optical flow displacement magnitude threshold (> 80.0 px/frame) resetting vectors.",
        metric_name="camera_flow_drift_px",
        unit="px",
    ),
    FailureCase(
        case_id="BALL_DROP_05",
        name="Ball Dropout Gap Bridging",
        category=FailureCategory.DETECTION,
        mode=FailureMode.BALL_DROPOUT,
        description="Small-object ball miss or motion blur fragmenting continuous possession.",
        root_cause="Low resolution footprint and rapid velocity causing detection dropouts for 1-2 frames.",
        mitigation_strategy="Temporal gap-filling buffer bridging missing ball intervals <= 2 frames.",
        metric_name="fragmented_intervals_count",
        unit="count",
    ),
]


class FailureCaseEvaluator:
    """Executes failure benchmark test cases and evaluates before/after mitigation efficacy."""

    def __init__(self, fps: float = 25.0):
        self.fps = fps

    def evaluate_track_swap(self) -> CaseEvaluationResult:
        """Case TS_PASS_01: Track-swap false pass rejection."""
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=4,
                start_time=0.0,
                end_time=0.16,
                player_id=1,
                team_id=1,
                duration_seconds=0.2,
                frame_count=5,
                termination_reason="pass",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=5,
                end_frame=10,
                start_time=0.2,
                end_time=0.4,
                player_id=2,
                team_id=1,
                duration_seconds=0.24,
                frame_count=6,
                termination_reason="end_of_clip",
            ),
        ]
        # Same physical player (displacement 0.45m in 1 frame)
        player_tracks = [
            {
                1: {"position_transformed": [30.0, 20.0]},
                2: {"position_transformed": [30.3, 20.3]},
            }
            for _ in range(12)
        ]

        raw_builder = EventBuilder(fps=self.fps, validate_kinematics=False)
        raw_events = raw_builder.build_events(intervals, player_tracks)
        raw_passes = sum(
            1 for e in raw_events if e.event_type == EventType.CANDIDATE_PASS.value
        )

        mitigated_builder = EventBuilder(fps=self.fps, validate_kinematics=True)
        mitigated_events = mitigated_builder.build_events(intervals, player_tracks)
        mitigated_passes = sum(
            1
            for e in mitigated_events
            if e.event_type == EventType.CANDIDATE_PASS.value
        )

        delta = float(mitigated_passes - raw_passes)
        improvement = 100.0 if raw_passes > 0 and mitigated_passes == 0 else 0.0

        return CaseEvaluationResult(
            case_id="TS_PASS_01",
            name="Track-Swap False Pass Rejection",
            category=FailureCategory.TRACKING.value,
            mode=FailureMode.TRACK_SWAP_PASS.value,
            metric_name="false_pass_count",
            unit="count",
            baseline_metric=float(raw_passes),
            mitigated_metric=float(mitigated_passes),
            delta=delta,
            improvement_percent=improvement,
            passed=mitigated_passes == 0,
            details={
                "baseline_events": [e.event_type for e in raw_events],
                "mitigated_events": [e.event_type for e in mitigated_events],
            },
        )

    def evaluate_superhuman_velocity(self) -> CaseEvaluationResult:
        """Case SH_VEL_02: Superhuman pass velocity rejection."""
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=3,
                start_time=0.0,
                end_time=0.12,
                player_id=1,
                team_id=1,
                duration_seconds=0.16,
                frame_count=4,
                termination_reason="pass",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=5,
                end_frame=10,
                start_time=0.2,
                end_time=0.4,
                player_id=2,
                team_id=1,
                duration_seconds=0.24,
                frame_count=6,
                termination_reason="end_of_clip",
            ),
        ]
        # Impossible jump: 70m in 2 frames (v = 875 m/s)
        player_tracks = [
            {
                1: {"position_transformed": [10.0, 10.0]},
                2: {"position_transformed": [80.0, 10.0]},
            }
            for _ in range(12)
        ]

        raw_builder = EventBuilder(
            fps=self.fps, validate_kinematics=False, maximum_pass_speed=10000.0
        )
        raw_events = raw_builder.build_events(intervals, player_tracks)
        raw_passes = sum(
            1 for e in raw_events if e.event_type == EventType.CANDIDATE_PASS.value
        )

        mitigated_builder = EventBuilder(
            fps=self.fps, validate_kinematics=True, maximum_pass_speed=45.0
        )
        mitigated_events = mitigated_builder.build_events(intervals, player_tracks)
        mitigated_passes = sum(
            1
            for e in mitigated_events
            if e.event_type == EventType.CANDIDATE_PASS.value
        )

        delta = float(mitigated_passes - raw_passes)
        improvement = 100.0 if raw_passes > 0 and mitigated_passes == 0 else 0.0

        return CaseEvaluationResult(
            case_id="SH_VEL_02",
            name="Superhuman Velocity Jump Filtering",
            category=FailureCategory.TRACKING.value,
            mode=FailureMode.SUPERHUMAN_VELOCITY.value,
            metric_name="superhuman_pass_count",
            unit="count",
            baseline_metric=float(raw_passes),
            mitigated_metric=float(mitigated_passes),
            delta=delta,
            improvement_percent=improvement,
            passed=mitigated_passes == 0,
            details={
                "baseline_events": [e.event_type for e in raw_events],
                "mitigated_events": [e.event_type for e in mitigated_events],
            },
        )

    def evaluate_possession_flicker(self) -> CaseEvaluationResult:
        """Case POSS_FLK_03: Scrum possession flicker suppression."""
        # Scrum: Player 1 (4 frames) -> Player 2 blip (1 frame) -> Player 1 (5 frames)
        player_tracks = [
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 2}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
        ]

        raw_extractor = PossessionIntervalExtractor(
            minimum_control_frames=1, fps=self.fps, hysteresis_frames=0
        )
        raw_intervals = raw_extractor.extract_intervals(player_tracks)
        raw_turnovers = sum(
            1 for i in raw_intervals if i.termination_reason == "turnover"
        )

        mitigated_extractor = PossessionIntervalExtractor(
            minimum_control_frames=1, fps=self.fps, hysteresis_frames=2
        )
        mitigated_intervals = mitigated_extractor.extract_intervals(player_tracks)
        mitigated_turnovers = sum(
            1 for i in mitigated_intervals if i.termination_reason == "turnover"
        )

        delta = float(mitigated_turnovers - raw_turnovers)
        improvement = 100.0 if raw_turnovers > 0 and mitigated_turnovers == 0 else 0.0

        return CaseEvaluationResult(
            case_id="POSS_FLK_03",
            name="Scrum Possession Flicker Suppression",
            category=FailureCategory.ANALYTICS.value,
            mode=FailureMode.POSSESSION_FLICKER.value,
            metric_name="false_turnover_count",
            unit="count",
            baseline_metric=float(raw_turnovers),
            mitigated_metric=float(mitigated_turnovers),
            delta=delta,
            improvement_percent=improvement,
            passed=mitigated_turnovers == 0 and len(mitigated_intervals) == 1,
            details={
                "baseline_intervals_count": len(raw_intervals),
                "mitigated_intervals_count": len(mitigated_intervals),
            },
        )

    def evaluate_camera_cut(self) -> CaseEvaluationResult:
        """Case CAM_CUT_04: Camera cut optical flow discontinuity detection."""
        # Generate 2 synthetic frames with high perimeter optical flow shift
        h, w = 720, 1280
        f0 = np.zeros((h, w, 3), dtype=np.uint8)
        f1 = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw clear border corner in vertical strip x in 0:20
        cv2.rectangle(f0, (5, 100), (15, 120), (255, 255, 255), -1)
        # In frame 1, simulate an artificial scene cut with a 35px vertical jump
        cv2.rectangle(f1, (5, 135), (15, 155), (255, 255, 255), -1)

        frames = [f0, f1]

        # Raw estimator with disabled cut threshold
        raw_est = CameraMotionEstimator(
            f0, minimum_distance=2.0, scene_cut_threshold=10000.0
        )
        raw_moves = raw_est.get_camera_movement(frames)
        raw_drift = (raw_moves[1][0] ** 2 + raw_moves[1][1] ** 2) ** 0.5

        # Mitigated estimator with cut threshold = 25.0 px
        mitigated_est = CameraMotionEstimator(
            f0, minimum_distance=2.0, scene_cut_threshold=25.0
        )
        mitigated_moves = mitigated_est.get_camera_movement(frames)
        mitigated_drift = (
            mitigated_moves[1][0] ** 2 + mitigated_moves[1][1] ** 2
        ) ** 0.5

        delta = float(mitigated_drift - raw_drift)
        improvement = 100.0 if raw_drift > 0 and mitigated_drift == 0.0 else 0.0

        return CaseEvaluationResult(
            case_id="CAM_CUT_04",
            name="Camera Cut Discontinuity Reset",
            category=FailureCategory.PERSPECTIVE_MOTION.value,
            mode=FailureMode.CAMERA_CUT_DISRUPTION.value,
            metric_name="camera_flow_drift_px",
            unit="px",
            baseline_metric=round(raw_drift, 2),
            mitigated_metric=round(mitigated_drift, 2),
            delta=round(delta, 2),
            improvement_percent=improvement,
            passed=mitigated_drift == 0.0,
            details={
                "raw_movement": raw_moves[1],
                "mitigated_movement": mitigated_moves[1],
            },
        )

    def evaluate_ball_dropout(self) -> CaseEvaluationResult:
        """Case BALL_DROP_05: Ball dropout gap bridging."""
        # Sequence: Player 1 (4 frames) -> ball missing for 2 frames -> Player 1 (4 frames)
        player_tracks = [
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": False, "team": 1}},  # frame 4: dropout
            {1: {"has_ball": False, "team": 1}},  # frame 5: dropout
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
        ]

        raw_extractor = PossessionIntervalExtractor(
            minimum_control_frames=1, fps=self.fps, max_gap_frames=0
        )
        raw_intervals = raw_extractor.extract_intervals(player_tracks)
        raw_count = len(raw_intervals)

        mitigated_extractor = PossessionIntervalExtractor(
            minimum_control_frames=1, fps=self.fps, max_gap_frames=2
        )
        mitigated_intervals = mitigated_extractor.extract_intervals(player_tracks)
        mitigated_count = len(mitigated_intervals)

        delta = float(mitigated_count - raw_count)
        improvement = 50.0 if raw_count == 2 and mitigated_count == 1 else 0.0

        return CaseEvaluationResult(
            case_id="BALL_DROP_05",
            name="Ball Dropout Gap Bridging",
            category=FailureCategory.DETECTION.value,
            mode=FailureMode.BALL_DROPOUT.value,
            metric_name="fragmented_intervals_count",
            unit="count",
            baseline_metric=float(raw_count),
            mitigated_metric=float(mitigated_count),
            delta=delta,
            improvement_percent=improvement,
            passed=mitigated_count == 1,
            details={
                "baseline_intervals": len(raw_intervals),
                "mitigated_intervals": len(mitigated_intervals),
            },
        )

    def run_suite(self) -> ErrorReport:
        """
        Execute all benchmark failure cases and assemble comprehensive report.

        Returns:
            ErrorReport containing results for all failure cases.
        """
        results: list[CaseEvaluationResult] = [
            self.evaluate_track_swap(),
            self.evaluate_superhuman_velocity(),
            self.evaluate_possession_flicker(),
            self.evaluate_camera_cut(),
            self.evaluate_ball_dropout(),
        ]

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = round((passed / total) * 100.0, 1) if total > 0 else 0.0

        return ErrorReport(
            timestamp=datetime.now(UTC).isoformat(),
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate_percent=pass_rate,
            results=results,
        )


class ErrorAnalysisExporter:
    """Exports diagnostic error analysis reports to JSON, CSV, and terminal tables."""

    @staticmethod
    def export_json(report: ErrorReport, output_path: Path | str) -> Path:
        """Write error report to JSON format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return path

    @staticmethod
    def export_csv(report: ErrorReport, output_path: Path | str) -> Path:
        """Write mitigation summary table to CSV format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case_id",
            "name",
            "category",
            "mode",
            "metric_name",
            "unit",
            "baseline_metric",
            "mitigated_metric",
            "delta",
            "improvement_percent",
            "passed",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in report.results:
                row = {
                    "case_id": res.case_id,
                    "name": res.name,
                    "category": res.category,
                    "mode": res.mode,
                    "metric_name": res.metric_name,
                    "unit": res.unit,
                    "baseline_metric": res.baseline_metric,
                    "mitigated_metric": res.mitigated_metric,
                    "delta": res.delta,
                    "improvement_percent": f"{res.improvement_percent:.1f}%",
                    "passed": res.passed,
                }
                writer.writerow(row)
        return path

    @staticmethod
    def format_terminal_table(report: ErrorReport) -> str:
        """Format a rich textual terminal table summarizing before/after metrics."""
        lines = [
            "",
            "=" * 92,
            "               FOOTBALL-CV RIGOROUS ERROR ANALYSIS & MITIGATION REPORT",
            "=" * 92,
            f" Timestamp: {report.timestamp}",
            f" Total Cases: {report.total_cases} | Passed: {report.passed_cases} | Failed: {report.failed_cases} | Pass Rate: {report.pass_rate_percent}%",
            "-" * 92,
            f" {'Case ID':<12} {'Category':<18} {'Baseline':<12} {'Mitigated':<12} {'Delta':<12} {'Status':<10}",
            "-" * 92,
        ]

        for r in report.results:
            status_str = "[PASS]" if r.passed else "[FAIL]"
            b_str = f"{r.baseline_metric} {r.unit}"
            m_str = f"{r.mitigated_metric} {r.unit}"
            d_str = f"{r.delta:+} ({r.improvement_percent:.0f}%)"
            lines.append(
                f" {r.case_id:<12} {r.category:<18} {b_str:<12} {m_str:<12} {d_str:<12} {status_str:<10}"
            )

        lines.append("=" * 92)
        return "\n".join(lines)
