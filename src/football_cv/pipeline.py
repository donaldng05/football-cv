"""
Decoupled end-to-end computer vision and analytics pipeline orchestrator.
"""

import logging
from contextlib import nullcontext
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
from .benchmark.profiler import PipelineProfiler
from .config import AppConfig
from .core import get_camera_motion_estimator, get_perspective_transformer
from .movement.speed_distance import SpeedDistanceEstimator
from .possession.assigner import PlayerBallAssigner
from .possession.events import PossessionInterval, PossessionIntervalExtractor
from .possession.interpolation import BallInterpolator, StreamingBallInterpolator
from .rendering.annotations import FrameAnnotator
from .rendering.video_writer import AnnotatedVideoWriter, IncrementalVideoWriter
from .teams.classifier import TeamClassifier
from .tracking import ObjectTracker, TrackSanitizer
from .utils.video import read_video, stream_video_chunks

logger = logging.getLogger("football_cv.pipeline")


class MatchPipeline:
    """Orchestrates detection, tracking, kinematics, and rendering for match analysis."""

    def __init__(self, config: AppConfig, profiler: PipelineProfiler | None = None):
        self.config = config
        self.profiler = profiler
        logger.info(f"Initializing MatchPipeline (backend='{config.vision.backend}')")

        self.tracker = ObjectTracker(
            model_path=config.model.path,
            confidence=config.model.confidence,
            batch_size=config.model.batch_size,
            device=config.model.device,
            engine=config.model.engine,
            track_activation_threshold=config.tracking.track_activation_threshold,
            lost_track_buffer=config.tracking.lost_track_buffer,
            minimum_matching_threshold=config.tracking.minimum_matching_threshold,
            frame_rate=int(config.video.frame_rate),
            ball_max_displacement_pixels=config.tracking.ball_tracker_max_distance,
            ball_min_confidence=config.tracking.ball_min_confidence,
            enable_reid_sanitizer=config.tracking.enable_reid_sanitizer,
            class_confidences=config.model.class_confidences,
        )
        self.view_transformer = get_perspective_transformer(
            pixel_vertices=config.perspective.pixel_vertices,
            court_width=config.perspective.court_width,
            court_length=config.perspective.court_length,
            out_of_bounds_policy=config.perspective.out_of_bounds_policy,
            backend=config.vision.backend,
        )
        self.ball_interpolator = BallInterpolator()
        self.speed_distance_estimator = SpeedDistanceEstimator(
            frame_window=config.movement.speed_window_frames,
            frame_rate=config.video.frame_rate,
            minimum_displacement=config.movement.minimum_displacement,
            method=config.movement.method,
            max_speed_kmh=getattr(config.movement, "max_speed_kmh", 38.0),
            enable_smoothing=getattr(config.movement, "enable_smoothing", True),
            min_speed_mps=getattr(config.movement, "min_speed_mps", 0.5),
        )
        self.team_classifier = TeamClassifier(
            color_space=config.team_classification.color_space,
            voting_window=config.team_classification.voting_window,
        )
        self.player_ball_assigner = PlayerBallAssigner(
            max_player_ball_distance=config.possession.max_player_ball_distance,
            max_player_ball_distance_meters=config.possession.max_player_ball_distance_meters,
            use_metric_distance=config.possession.use_metric_distance,
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

    def _time_stage(self, stage_name: str):
        """Return context manager for timing a stage if profiling is active."""
        if self.profiler is not None:
            return self.profiler.time_stage(stage_name)
        return nullcontext()

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
        with self._time_stage("detection_and_tracking"):
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

        # 2. Ball position interpolation (interpolate missing ball detections before calculating spatial positions)
        with self._time_stage("ball_interpolation"):
            tracks["balls"] = self.ball_interpolator.interpolate_ball_positions(
                tracks["balls"],
                limit=self.config.possession.maximum_missing_ball_frames,
                method=self.config.possession.ball_interpolation_method,
            )

        self.tracker.add_positions_to_tracks(tracks)

        # 3. Camera movement compensation
        with self._time_stage("camera_motion"):
            cam_estimator = get_camera_motion_estimator(
                frames[0],
                minimum_distance=self.config.camera_motion.minimum_distance,
                scene_cut_threshold=self.config.camera_motion.scene_cut_threshold,
                margin_ratio_x=self.config.camera_motion.margin_ratio_x,
                margin_ratio_y=self.config.camera_motion.margin_ratio_y,
                use_dynamic_margins=self.config.camera_motion.use_dynamic_margins,
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

            cam_estimator.add_adjust_positions_to_tracks(
                tracks, camera_movement, cumulative=True
            )

        # 4. Perspective transformation (homography projection to real-world meters)
        with self._time_stage("perspective_transform"):
            camera_matrices = getattr(cam_estimator, "camera_matrices", None)
            if camera_matrices and len(camera_matrices) > len(frames):
                camera_matrices = camera_matrices[start_idx:end_idx]
            self.view_transformer.add_transformed_position_to_tracks(
                tracks, camera_matrices=camera_matrices
            )

        # 5. Speed and distance estimation
        with self._time_stage("speed_distance"):
            self.speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # 6. Team color assignment
        with self._time_stage("team_assignment"):
            if tracks["players"] and any(len(p) > 0 for p in tracks["players"]):
                self.team_classifier.assign_team_color(
                    frames,
                    tracks["players"],
                    sample_frames=self.config.team_classification.sample_frames,
                    min_box_area=self.config.team_classification.min_box_area,
                )

                for frame_num, player_tracks in enumerate(tracks["players"]):
                    for player_id, track in player_tracks.items():
                        team = self.team_classifier.get_player_team(
                            frames[frame_num],
                            track["bbox"],
                            player_id,
                            role=track.get("role", "player"),
                            frame_players=player_tracks,
                        )
                        tracks["players"][frame_num][player_id]["team"] = team
                        tracks["players"][frame_num][player_id]["team_color"] = (
                            self.team_classifier.team_colors.get(team, (0, 0, 255))
                        )

                if self.config.tracking.enable_reid_sanitizer:
                    TrackSanitizer.sanitize_team_consistency(tracks["players"])

        # 7. Ball possession assignment (metric space proximity with pixel fallback)
        team_ball_control = []
        with self._time_stage("possession_assignment"):
            for frame_num, player_track in enumerate(tracks["players"]):
                ball_dict = (
                    tracks["balls"][frame_num]
                    if frame_num < len(tracks["balls"])
                    else {}
                )
                ball_info = ball_dict.get(1, {})
                ball_bbox = ball_info.get("bbox", [])
                ball_transformed = ball_info.get("position_transformed")

                assigned_player = self.player_ball_assigner.assign_ball_to_player(
                    player_track, ball_bbox, ball_transformed=ball_transformed
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
        with self._time_stage("analytics_inference"):
            if self.config.analytics.enabled:
                intervals = self.interval_extractor.extract_intervals(
                    player_tracks=tracks["players"],
                    team_ball_control=team_ball_control,
                )
                events = self.event_builder.build_events(
                    intervals=intervals,
                    player_tracks=tracks["players"],
                    ball_tracks=tracks.get("balls"),
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
        with self._time_stage("rendering"):
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

    def _process_chunk_analysis(
        self,
        chunk_frames: list[np.ndarray],
        is_last: bool,
        cam_estimator_holder: dict[str, Any],
        streaming_interpolator: StreamingBallInterpolator,
        all_tracks: dict[str, list[dict[int, Any]]],
        all_camera_movement: list[tuple[float, float]],
        team_ball_control: list[int],
    ) -> dict[str, Any]:
        """
        Analyze a chunk of frames (detection, tracking, homography, team assignment, ball possession)
        without rendering, accumulating metadata into all_tracks.
        """
        if not chunk_frames:
            return {"players": [], "referees": [], "balls": []}

        # 1. Multi-object tracking for chunk
        with self._time_stage("detection_and_tracking"):
            chunk_tracks = self.tracker.track_chunk(chunk_frames)

        # 2. Camera motion estimation
        with self._time_stage("camera_motion"):
            if cam_estimator_holder["estimator"] is None:
                cam_estimator = get_camera_motion_estimator(
                    chunk_frames[0],
                    minimum_distance=self.config.camera_motion.minimum_distance,
                    scene_cut_threshold=self.config.camera_motion.scene_cut_threshold,
                    margin_ratio_x=self.config.camera_motion.margin_ratio_x,
                    margin_ratio_y=self.config.camera_motion.margin_ratio_y,
                    use_dynamic_margins=self.config.camera_motion.use_dynamic_margins,
                    backend=self.config.vision.backend,
                )
                cam_estimator_holder["estimator"] = cam_estimator
                self._streaming_cam_estimator = cam_estimator
            else:
                cam_estimator = cam_estimator_holder["estimator"]

            chunk_movement = cam_estimator.estimate_chunk(chunk_frames)
            cam_estimator.add_adjust_positions_to_chunk(chunk_tracks, chunk_movement)
            chunk_matrices = getattr(cam_estimator, "last_chunk_matrices", None)

        # 3. Perspective transformation (homography)
        with self._time_stage("perspective_transform"):
            self.view_transformer.add_transformed_position_to_tracks(
                chunk_tracks, camera_matrices=chunk_matrices
            )

        # 4. Speed & distance estimation
        with self._time_stage("speed_distance"):
            self.speed_distance_estimator.estimate_chunk(chunk_tracks)

        # 5. Team classification
        with self._time_stage("team_assignment"):
            if (
                self.team_classifier.kmeans is None
                and chunk_tracks["players"]
                and any(len(p) > 0 for p in chunk_tracks["players"])
            ):
                self.team_classifier.assign_team_color(
                    chunk_frames,
                    chunk_tracks["players"],
                    sample_frames=self.config.team_classification.sample_frames,
                    min_box_area=self.config.team_classification.min_box_area,
                )

            for frame_num, player_tracks in enumerate(chunk_tracks["players"]):
                for player_id, track in player_tracks.items():
                    team = self.team_classifier.get_player_team(
                        chunk_frames[frame_num],
                        track["bbox"],
                        player_id,
                        role=track.get("role", "player"),
                        frame_players=player_tracks,
                    )
                    track["team"] = team
                    track["team_color"] = self.team_classifier.team_colors.get(
                        team, (0, 0, 255)
                    )

        # 6. Ball interpolation for chunk
        with self._time_stage("ball_interpolation"):
            interpolated_chunk_balls = streaming_interpolator.update(
                chunk_tracks["balls"], is_last_chunk=is_last
            )
            if len(interpolated_chunk_balls) == len(chunk_tracks["balls"]):
                chunk_tracks["balls"] = interpolated_chunk_balls

        # 7. Ball possession assignment
        with self._time_stage("possession_assignment"):
            chunk_control: list[int] = []
            for frame_num, player_track in enumerate(chunk_tracks["players"]):
                ball_dict = (
                    chunk_tracks["balls"][frame_num]
                    if frame_num < len(chunk_tracks["balls"])
                    else {}
                )
                ball_info = ball_dict.get(1, {})
                ball_bbox = ball_info.get("bbox", [])
                ball_transformed = ball_info.get("position_transformed")

                assigned_player = self.player_ball_assigner.assign_ball_to_player(
                    player_track, ball_bbox, ball_transformed=ball_transformed
                )
                if assigned_player != -1:
                    chunk_tracks["players"][frame_num][assigned_player]["has_ball"] = (
                        True
                    )
                    team = chunk_tracks["players"][frame_num][assigned_player].get(
                        "team", 1
                    )
                    chunk_control.append(team)
                else:
                    last_team = (
                        chunk_control[-1]
                        if chunk_control
                        else (team_ball_control[-1] if team_ball_control else 1)
                    )
                    chunk_control.append(last_team)

            team_ball_control.extend(chunk_control)

        # Retain lightweight metadata
        all_tracks["players"].extend(chunk_tracks["players"])
        all_tracks["referees"].extend(chunk_tracks["referees"])
        all_tracks["balls"].extend(chunk_tracks["balls"])
        all_camera_movement.extend(chunk_movement)
        return chunk_tracks

    def _process_and_render_chunk(
        self,
        chunk_frames: list[np.ndarray],
        is_last: bool,
        cam_estimator_holder: dict[str, Any],
        streaming_interpolator: StreamingBallInterpolator,
        writer: IncrementalVideoWriter | None,
        all_tracks: dict[str, list[dict[int, Any]]],
        all_camera_movement: list[tuple[float, float]],
        team_ball_control: list[int],
    ) -> None:
        """Backward-compatible helper that processes chunk analysis and optionally renders."""
        start_count = len(all_tracks["players"])
        chunk_tracks = self._process_chunk_analysis(
            chunk_frames=chunk_frames,
            is_last=is_last,
            cam_estimator_holder=cam_estimator_holder,
            streaming_interpolator=streaming_interpolator,
            all_tracks=all_tracks,
            all_camera_movement=all_camera_movement,
            team_ball_control=team_ball_control,
        )
        if writer is not None:
            end_count = len(all_tracks["players"])
            chunk_movement = all_camera_movement[start_count:end_count]
            chunk_control = np.array(team_ball_control[start_count:end_count])
            with self._time_stage("rendering"):
                annotated_chunk = self.annotator.draw_annotations(
                    chunk_frames, chunk_tracks, chunk_control
                )
                annotated_chunk = self.annotator.draw_camera_movement(
                    annotated_chunk, chunk_movement
                )
                annotated_chunk = self.annotator.draw_speed_and_distance(
                    annotated_chunk, chunk_tracks
                )
                writer.write_chunk(annotated_chunk)

    def run_streaming(self, profile: bool = False) -> dict[str, Any]:
        """
        Execute memory-bounded sliding-window streaming pipeline.
        Peak memory consumption is strictly bounded to chunk_size frames (~400 MB),
        enabling full match processing without OOM crashes while guaranteeing that
        the rendered video matches the finalized sanitized analytics.
        """
        if profile and self.profiler is None:
            self.profiler = PipelineProfiler()

        logger.info(
            f"Starting streaming execution (chunk_size={self.config.streaming.chunk_size})"
        )
        video_path = self.config.video.input_path
        out_path = self.config.video.output_path
        chunk_size = self.config.streaming.chunk_size
        start_frame = self.config.video.start_frame
        end_frame = self.config.video.end_frame

        all_tracks: dict[str, list[dict[int, Any]]] = {
            "players": [],
            "referees": [],
            "balls": [],
        }
        all_camera_movement: list[tuple[float, float]] = []
        team_ball_control: list[int] = []

        self.speed_distance_estimator.reset()
        streaming_interpolator = StreamingBallInterpolator(
            limit=self.config.possession.maximum_missing_ball_frames,
            method=self.config.possession.ball_interpolation_method,
        )

        chunk_gen = stream_video_chunks(
            video_path,
            chunk_size=chunk_size,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        try:
            prev_chunk_info = next(chunk_gen)
        except StopIteration:
            logger.warning("No video frames available to stream")
            return {
                "tracks": all_tracks,
                "camera_movement": all_camera_movement,
                "team_ball_control": np.array([]),
                "possession_intervals": [],
                "events": [],
            }

        # Pass 1: Chunked extraction and tracking (memory strictly bounded to chunk_size frames)
        cam_holder: dict[str, Any] = {"estimator": None}
        for next_chunk_info in chunk_gen:
            self._process_chunk_analysis(
                prev_chunk_info[1],
                is_last=False,
                cam_estimator_holder=cam_holder,
                streaming_interpolator=streaming_interpolator,
                all_tracks=all_tracks,
                all_camera_movement=all_camera_movement,
                team_ball_control=team_ball_control,
            )
            prev_chunk_info = next_chunk_info

        self._process_chunk_analysis(
            prev_chunk_info[1],
            is_last=True,
            cam_estimator_holder=cam_holder,
            streaming_interpolator=streaming_interpolator,
            all_tracks=all_tracks,
            all_camera_movement=all_camera_movement,
            team_ball_control=team_ball_control,
        )

        # Global Track Sanitization & Kinematics before rendering!
        if self.config.tracking.enable_reid_sanitizer and all_tracks["players"]:
            TrackSanitizer.sanitize_team_consistency(all_tracks["players"])
            TrackSanitizer.stitch_fragmented_tracks(all_tracks["players"])

        self.speed_distance_estimator.add_speed_and_distance_to_tracks(all_tracks)

        # Pass 2: Chunked rendering to disk (memory strictly bounded to chunk_size)
        with IncrementalVideoWriter(
            output_path=out_path,
            fps=self.config.video.frame_rate,
        ) as writer:
            render_gen = stream_video_chunks(
                video_path,
                chunk_size=chunk_size,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            for start_idx, chunk_frames in render_gen:
                end_idx = start_idx + len(chunk_frames)
                chunk_tracks = {k: all_tracks[k][start_idx:end_idx] for k in all_tracks}
                chunk_movement = all_camera_movement[start_idx:end_idx]
                chunk_control = np.array(team_ball_control[start_idx:end_idx])

                with self._time_stage("rendering"):
                    annotated_chunk = self.annotator.draw_annotations(
                        chunk_frames, chunk_tracks, chunk_control
                    )
                    annotated_chunk = self.annotator.draw_camera_movement(
                        annotated_chunk, chunk_movement
                    )
                    annotated_chunk = self.annotator.draw_speed_and_distance(
                        annotated_chunk, chunk_tracks
                    )
                    writer.write_chunk(annotated_chunk)

        intervals: list[PossessionInterval] = []
        events: list[CandidateEvent] = []
        if self.config.analytics.enabled:
            intervals = self.interval_extractor.extract_intervals(
                player_tracks=all_tracks["players"],
                team_ball_control=team_ball_control,
            )
            events = self.event_builder.build_events(
                intervals=intervals,
                player_tracks=all_tracks["players"],
                ball_tracks=all_tracks["balls"],
            )

        results = {
            "tracks": all_tracks,
            "camera_movement": all_camera_movement,
            "team_ball_control": np.array(team_ball_control),
            "possession_intervals": intervals,
            "events": events,
        }

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

        if self.profiler is not None:
            total_frames = len(all_tracks["players"])
            summary = self.profiler.summarize(num_frames=total_frames)
            results["profile_summary"] = summary.to_dict()
            results["profile_summary_obj"] = summary

        return results

    def run(self, profile: bool = False) -> dict[str, Any]:
        """Load video, execute full analysis pipeline, and write annotated video."""
        if self.config.streaming.enabled:
            return self.run_streaming(profile=profile)

        if profile and self.profiler is None:
            self.profiler = PipelineProfiler()

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

        if self.profiler is not None:
            summary = self.profiler.summarize(num_frames=len(frames))
            results["profile_summary"] = summary.to_dict()
            results["profile_summary_obj"] = summary

        return results
