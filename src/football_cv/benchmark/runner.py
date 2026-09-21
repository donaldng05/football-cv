"""
Benchmark runner orchestrating pipeline profiling and configuration sweeps.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..exceptions import BenchmarkError
from ..pipeline import MatchPipeline
from ..utils.video import read_video
from .environment import EnvironmentCollector
from .profiler import PipelineProfiler, ProfileSummary

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRunResult:
    """Outcome and latency profile of a single benchmark run."""

    run_id: str
    run_name: str
    model_path: str
    confidence: float
    batch_size: int
    device: str
    num_frames: int
    profile: ProfileSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "model_path": self.model_path,
            "confidence": self.confidence,
            "batch_size": self.batch_size,
            "device": self.device,
            "num_frames": self.num_frames,
            "profile": self.profile.to_dict(),
        }


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report containing environment metadata and all sweep runs."""

    benchmark_id: str
    created_at: str
    environment: dict[str, Any]
    runs: list[BenchmarkRunResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "created_at": self.created_at,
            "environment": self.environment,
            "runs": [r.to_dict() for r in self.runs],
        }


class BenchmarkRunner:
    """
    Executes automated pipeline profiling passes and sweeps across model checkpoints
    and confidence thresholds.
    """

    def __init__(
        self,
        base_config: AppConfig | None = None,
        warmup_frames: int = 5,
    ):
        self.base_config = base_config
        self.warmup_frames = max(0, warmup_frames)

    def profile_pipeline(
        self,
        pipeline: MatchPipeline,
        frames: list[Any],
        profiler: PipelineProfiler,
    ) -> dict[str, Any]:
        """
        Execute pipeline while instrumenting each computational phase.

        Returns:
            Dictionary containing final pipeline results.
        """
        if not frames:
            return {}

        # 1. Multi-object detection and tracking
        with profiler.time_stage("detection_and_tracking"):
            use_stubs = pipeline.config.tracking.use_cached_tracks
            stub_path = pipeline.config.tracking.cache_path
            tracks = pipeline.tracker.get_tracked_objects(
                frames, read_from_stub=use_stubs, stub_path=stub_path
            )

            start_idx = pipeline.config.video.start_frame
            end_idx = start_idx + len(frames)
            if len(tracks["players"]) > len(frames):
                tracks["players"] = tracks["players"][start_idx:end_idx]
                tracks["referees"] = tracks["referees"][start_idx:end_idx]
                tracks["balls"] = tracks["balls"][start_idx:end_idx]
            while len(tracks["players"]) < len(frames):
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["balls"].append({})

            pipeline.tracker.add_positions_to_tracks(tracks)

        # 2. Camera motion compensation
        with profiler.time_stage("camera_motion"):
            from ..core import get_camera_motion_estimator

            cam_estimator = get_camera_motion_estimator(
                frames[0],
                backend=pipeline.config.vision.backend,
            )
            cam_stub = pipeline.config.tracking.camera_movement_cache_path
            camera_movement = cam_estimator.get_camera_movement(
                frames, read_from_stub=use_stubs, stub_path=cam_stub
            )
            if len(camera_movement) > len(frames):
                camera_movement = camera_movement[start_idx:end_idx]
            while len(camera_movement) < len(frames):
                camera_movement.append((0.0, 0.0))

            cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

        # 3. Perspective homography
        with profiler.time_stage("perspective_transform"):
            pipeline.view_transformer.add_transformed_position_to_tracks(tracks)

        # 4. Ball position interpolation
        with profiler.time_stage("ball_interpolation"):
            tracks["balls"] = pipeline.ball_interpolator.interpolate_ball_positions(
                tracks["balls"],
                limit=pipeline.config.possession.maximum_missing_ball_frames,
            )

        # 5. Speed and distance calculation
        with profiler.time_stage("speed_distance"):
            pipeline.speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # 6. Team color classification
        with profiler.time_stage("team_assignment"):
            if tracks["players"] and len(tracks["players"][0]) > 0:
                pipeline.team_classifier.assign_team_color(
                    frames[0], tracks["players"][0]
                )
                for frame_num, player_tracks in enumerate(tracks["players"]):
                    for player_id, track in player_tracks.items():
                        team = pipeline.team_classifier.get_player_team(
                            frames[frame_num], track["bbox"], player_id
                        )
                        tracks["players"][frame_num][player_id]["team"] = team
                        tracks["players"][frame_num][player_id]["team_color"] = (
                            pipeline.team_classifier.team_colors.get(team, (0, 0, 255))
                        )

        # 7. Ball possession assignment
        with profiler.time_stage("possession_assignment"):
            team_ball_control: list[int] = []
            for frame_num, player_track in enumerate(tracks["players"]):
                ball_dict = (
                    tracks["balls"][frame_num]
                    if frame_num < len(tracks["balls"])
                    else {}
                )
                ball_bbox = ball_dict.get(1, {}).get("bbox", [])
                assigned_player = pipeline.player_ball_assigner.assign_ball_to_player(
                    player_track, ball_bbox
                )

                if assigned_player != -1:
                    tracks["players"][frame_num][assigned_player]["has_ball"] = True
                    team = tracks["players"][frame_num][assigned_player].get("team", 1)
                    team_ball_control.append(team)
                else:
                    last_team = team_ball_control[-1] if team_ball_control else 1
                    team_ball_control.append(last_team)

        # 8. Event and possession inference
        with profiler.time_stage("analytics_inference"):
            intervals = pipeline.interval_extractor.extract_intervals(
                player_tracks=tracks["players"],
                team_ball_control=team_ball_control,
            )
            events = pipeline.event_builder.build_events(
                intervals=intervals,
                player_tracks=tracks["players"],
            )

        # 9. Frame rendering
        with profiler.time_stage("rendering"):
            annotated_frames = pipeline.annotator.draw_annotations(
                frames, tracks, team_ball_control
            )
            annotated_frames = pipeline.annotator.draw_camera_movement(
                annotated_frames, camera_movement
            )
            annotated_frames = pipeline.annotator.draw_speed_and_distance(
                annotated_frames, tracks
            )

        return {
            "tracks": tracks,
            "camera_movement": camera_movement,
            "team_ball_control": team_ball_control,
            "possession_intervals": intervals,
            "events": events,
            "annotated_frames": annotated_frames,
        }

    def run_single(
        self,
        config: AppConfig,
        num_frames: int = 50,
        run_name: str = "benchmark_run",
    ) -> BenchmarkRunResult:
        """
        Execute a single profiled benchmarking run on a configured slice of frames.

        Args:
            config: Validated AppConfig for the run.
            num_frames: Number of frames to benchmark.
            run_name: Human-readable identifier for this run.

        Returns:
            BenchmarkRunResult containing stage latencies and throughput metrics.
        """
        video_path = config.video.input_path
        if not Path(video_path).is_file():
            raise BenchmarkError(f"Input video file not found: {video_path}")

        profiler = PipelineProfiler()

        # Measure video decoding
        total_frames_needed = self.warmup_frames + num_frames
        with profiler.time_stage("video_decoding"):
            all_frames = read_video(
                video_path,
                start_frame=config.video.start_frame,
                end_frame=config.video.start_frame + total_frames_needed,
            )

        if not all_frames:
            raise BenchmarkError(f"Could not read video frames from {video_path}")

        # Split warmup vs profiled frames
        warmup = all_frames[: self.warmup_frames] if self.warmup_frames > 0 else []
        test_frames = all_frames[self.warmup_frames : total_frames_needed]

        pipeline = MatchPipeline(config)

        # Warmup pass (if enabled)
        if warmup:
            logger.info(f"Executing warmup on {len(warmup)} frames...")
            dummy_profiler = PipelineProfiler()
            self.profile_pipeline(pipeline, warmup, dummy_profiler)

        # Profiled test pass
        logger.info(f"Executing benchmark profiling on {len(test_frames)} frames...")
        self.profile_pipeline(pipeline, test_frames, profiler)

        summary = profiler.summarize(num_frames=len(test_frames))

        return BenchmarkRunResult(
            run_id=str(uuid.uuid4())[:8],
            run_name=run_name,
            model_path=config.model.path,
            confidence=config.model.confidence,
            batch_size=config.model.batch_size,
            device=config.model.device,
            num_frames=len(test_frames),
            profile=summary,
        )

    def run_sweep(
        self,
        config: AppConfig,
        num_frames: int = 50,
        confidences: list[float] | None = None,
        models: list[str] | None = None,
    ) -> BenchmarkReport:
        """
        Execute automated sweep across confidence thresholds and model variants.

        Args:
            config: Baseline AppConfig.
            num_frames: Number of frames to benchmark per configuration.
            confidences: List of detection confidence thresholds to test.
            models: List of model checkpoint paths to evaluate.

        Returns:
            BenchmarkReport containing environment metadata and all run profiles.
        """
        benchmark_id = f"bench_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        env = EnvironmentCollector.collect()

        conf_list = (
            confidences if confidences is not None else [config.model.confidence]
        )
        model_list = models if models is not None else [config.model.path]

        runs: list[BenchmarkRunResult] = []

        for m_path in model_list:
            for conf in conf_list:
                run_name = f"{Path(m_path).stem}_conf_{conf:.2f}"
                logger.info(f"Starting sweep run: {run_name}")

                # Clone config with overrides
                cfg = AppConfig(
                    model=config.model.__class__(
                        path=m_path,
                        confidence=conf,
                        device=config.model.device,
                        batch_size=config.model.batch_size,
                    ),
                    video=config.video,
                    tracking=config.tracking,
                    possession=config.possession,
                    movement=config.movement,
                    perspective=config.perspective,
                    analytics=config.analytics,
                    vision=config.vision,
                    logging=config.logging,
                )
                cfg.validate()

                run_res = self.run_single(
                    config=cfg,
                    num_frames=num_frames,
                    run_name=run_name,
                )
                runs.append(run_res)

        return BenchmarkReport(
            benchmark_id=benchmark_id,
            created_at=datetime.now(UTC).isoformat(),
            environment=env,
            runs=runs,
        )
