"""
Decoupled end-to-end computer vision and analytics pipeline orchestrator.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .analytics import (
    AnalyticsExporter,
    CandidateEvent,
    EventBuilder,
    HeatmapGenerator,
    PassNetworkGenerator,
)
from .config import AppConfig
from .core import get_camera_motion_estimator, get_perspective_transformer
from .movement.speed_distance import SpeedDistanceEstimator
from .possession.assigner import PlayerBallAssigner
from .possession.events import PossessionInterval, PossessionIntervalExtractor
from .possession.interpolation import BallInterpolator
from .rendering.annotations import FrameAnnotator
from .rendering.video_writer import AnnotatedVideoWriter
from .teams.classifier import TeamClassifier
from .tracking.tracker import ObjectTracker
from .utils.video import read_video

logger = logging.getLogger("football_cv.pipeline")


class MatchPipeline:
    """Orchestrates detection, tracking, kinematics, and rendering for match analysis."""

    def __init__(self, config: AppConfig):
        self.config = config
        logger.info(f"Initializing MatchPipeline (backend='{config.vision.backend}')")

        self.tracker = ObjectTracker(
            model_path=config.model.path,
            confidence=config.model.confidence,
            batch_size=config.model.batch_size,
            device=config.model.device,
        )
        self.view_transformer = get_perspective_transformer(
            pixel_vertices=config.perspective.pixel_vertices,
            court_width=config.perspective.court_width,
            court_length=config.perspective.court_length,
            backend=config.vision.backend,
        )
        self.ball_interpolator = BallInterpolator()
        self.speed_distance_estimator = SpeedDistanceEstimator(
            frame_window=config.movement.speed_window_frames,
            frame_rate=config.video.frame_rate,
            minimum_displacement=config.movement.minimum_displacement,
        )
        self.team_classifier = TeamClassifier()
        self.player_ball_assigner = PlayerBallAssigner(
            max_player_ball_distance=config.possession.max_player_ball_distance
        )
        self.annotator = FrameAnnotator()

        self.interval_extractor = PossessionIntervalExtractor(
            minimum_control_frames=config.possession.minimum_control_frames,
            fps=config.video.frame_rate,
        )
        self.event_builder = EventBuilder(
            maximum_transition_frames=config.analytics.pass_network.maximum_transition_frames,
            fps=config.video.frame_rate,
        )
        self.exporter = AnalyticsExporter(
            output_dir=config.analytics.export_dir,
            fps=config.video.frame_rate,
        )

    def process(
        self, frames: list[np.ndarray], tracks_stub_path: str | None = None
    ) -> dict[str, Any]:
        """
        Execute core computer vision and analytics tracking computation (no video writing).

        Returns:
            Dictionary containing processed tracks, camera movement, and team ball control stats.
        """
        logger.info(f"Processing {len(frames)} video frames")

        # 1. Multi-object tracking
        use_stubs = self.config.tracking.use_cached_tracks
        stub_path = tracks_stub_path or self.config.tracking.cache_path
        tracks = self.tracker.get_tracked_objects(
            frames, read_from_stub=use_stubs, stub_path=stub_path
        )

        # Align stub frame counts with input frames slice if loaded from full-match cache
        start_idx = self.config.video.start_frame
        end_idx = start_idx + len(frames)
        if len(tracks["players"]) > len(frames):
            tracks["players"] = tracks["players"][start_idx:end_idx]
            tracks["referees"] = tracks["referees"][start_idx:end_idx]
            tracks["balls"] = tracks["balls"][start_idx:end_idx]
        while len(tracks["players"]) < len(frames):
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["balls"].append({})

        self.tracker.add_positions_to_tracks(tracks)

        # 2. Camera movement compensation
        cam_estimator = get_camera_motion_estimator(
            frames[0],
            backend=self.config.vision.backend,
        )
        cam_stub = self.config.tracking.camera_movement_cache_path
        camera_movement = cam_estimator.get_camera_movement(
            frames, read_from_stub=use_stubs, stub_path=cam_stub
        )
        if len(camera_movement) > len(frames):
            camera_movement = camera_movement[start_idx:end_idx]
        while len(camera_movement) < len(frames):
            camera_movement.append((0.0, 0.0))

        cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

        # 3. Perspective transformation
        self.view_transformer.add_transformed_position_to_tracks(tracks)

        # 4. Ball position interpolation
        tracks["balls"] = self.ball_interpolator.interpolate_ball_positions(
            tracks["balls"], limit=self.config.possession.maximum_missing_ball_frames
        )

        # 5. Speed and distance estimation
        self.speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # 6. Team color assignment
        if tracks["players"] and len(tracks["players"][0]) > 0:
            self.team_classifier.assign_team_color(frames[0], tracks["players"][0])

            for frame_num, player_tracks in enumerate(tracks["players"]):
                for player_id, track in player_tracks.items():
                    team = self.team_classifier.get_player_team(
                        frames[frame_num], track["bbox"], player_id
                    )
                    tracks["players"][frame_num][player_id]["team"] = team
                    tracks["players"][frame_num][player_id]["team_color"] = (
                        self.team_classifier.team_colors.get(team, (0, 0, 255))
                    )

        # 7. Ball possession assignment
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks["players"]):
            ball_dict = (
                tracks["balls"][frame_num] if frame_num < len(tracks["balls"]) else {}
            )
            ball_bbox = ball_dict.get(1, {}).get("bbox", [])

            assigned_player = self.player_ball_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team = tracks["players"][frame_num][assigned_player].get("team", 1)
                team_ball_control.append(team)
            else:
                last_team = team_ball_control[-1] if team_ball_control else 1
                team_ball_control.append(last_team)

        # 8. Possession intervals and candidate events
        intervals: list[PossessionInterval] = []
        events: list[CandidateEvent] = []
        if self.config.analytics.enabled:
            intervals = self.interval_extractor.extract_intervals(
                player_tracks=tracks["players"],
                team_ball_control=team_ball_control,
            )
            events = self.event_builder.build_events(
                intervals=intervals,
                player_tracks=tracks["players"],
            )

        return {
            "tracks": tracks,
            "camera_movement": camera_movement,
            "team_ball_control": np.array(team_ball_control),
            "possession_intervals": intervals,
            "events": events,
        }

    def render(
        self,
        frames: list[np.ndarray],
        tracks: dict[str, Any],
        camera_movement: list[Any],
        team_ball_control: np.ndarray,
        output_path: str | None = None,
    ) -> list[np.ndarray]:
        """Render annotation overlays and save the output video."""
        logger.info("Rendering visual annotations")
        annotated_frames = self.annotator.draw_annotations(
            frames, tracks, team_ball_control
        )
        annotated_frames = self.annotator.draw_camera_movement(
            annotated_frames, camera_movement
        )
        annotated_frames = self.annotator.draw_speed_and_distance(
            annotated_frames, tracks
        )

        out_path = output_path or self.config.video.output_path
        if out_path:
            logger.info(f"Saving annotated video to {out_path}")
            writer = AnnotatedVideoWriter(
                output_path=out_path, fps=self.config.video.frame_rate
            )
            writer.write_frames(annotated_frames)

        return annotated_frames

    def run(self) -> dict[str, Any]:
        """Load video, execute full analysis pipeline, and write annotated video."""
        logger.info(f"Reading video from {self.config.video.input_path}")
        frames = read_video(
            self.config.video.input_path,
            start_frame=self.config.video.start_frame,
            end_frame=self.config.video.end_frame,
        )

        results = self.process(frames)
        self.render(
            frames=frames,
            tracks=results["tracks"],
            camera_movement=results["camera_movement"],
            team_ball_control=results["team_ball_control"],
        )

        if self.config.analytics.enabled and self.config.analytics.export_events:
            logger.info(
                f"Exporting structured match data to {self.config.analytics.export_dir}"
            )
            export_paths = self.exporter.export_all(
                tracks=results["tracks"],
                intervals=results["possession_intervals"],
                events=results["events"],
                config=self.config,
            )
            results["export_paths"] = export_paths

        if self.config.analytics.enabled and self.config.analytics.heatmap.enabled:
            logger.info("Generating possession heatmaps")
            report_dir = Path(self.config.analytics.export_dir).parent / "report"
            heatmap_gen = HeatmapGenerator(config=self.config, output_dir=report_dir)
            if (
                "export_paths" in results
                and "player_tracking_json" in results["export_paths"]
            ):
                results["heatmap_paths"] = heatmap_gen.generate_from_file(
                    results["export_paths"]["player_tracking_json"]
                )

        if self.config.analytics.enabled and self.config.analytics.pass_network.enabled:
            logger.info("Generating tactical pass networks")
            report_dir = Path(self.config.analytics.export_dir).parent / "report"
            pass_gen = PassNetworkGenerator(config=self.config, output_dir=report_dir)
            results["pass_network_paths"] = pass_gen.generate(
                events=results.get("events", []),
                intervals=results.get("possession_intervals", []),
            )

        return results
