"""
Unit tests for kinematics smoothing, event trajectory alignment,
streaming synchronization, and zero-copy annotator fixes.
"""

import numpy as np

from football_cv.analytics.event_builder import EventBuilder, EventType
from football_cv.movement.speed_distance import SpeedDistanceEstimator
from football_cv.possession.events import PossessionInterval
from football_cv.rendering.annotations import FrameAnnotator


class TestKinematicSmoothingAndJitter:
    """Test suite ensuring metric jitter does not create supersonic speeds or marathon distances."""

    def test_stationary_player_jitter_does_not_accumulate_distance(self):
        estimator = SpeedDistanceEstimator(
            frame_window=5,
            frame_rate=25.0,
            minimum_displacement=0.05,
            enable_smoothing=True,
            min_speed_mps=0.5,
            method="rolling",
        )

        num_frames = 100
        tracks = {"players": []}
        # A player stationary at (50m, 20m) experiencing +-0.5m perspective projection jitter
        for f in range(num_frames):
            jitter = 0.5 if (f % 2 == 0) else -0.5
            tracks["players"].append(
                {1: {"position_transformed": [50.0 + jitter, 20.0]}}
            )

        estimator.add_speed_and_distance_to_tracks(tracks)

        # Without filtering, 100 frames of 1.0m step distance would accumulate 99 meters!
        # With Savitzky-Golay smoothing and stationary deadband (0.5 m/s):
        final_distance = tracks["players"][-1][1]["distance_covered"]
        assert final_distance < 1.0, f"Stationary player accumulated {final_distance}m"

        # Speeds should be 0.0 km/h (deadband active)
        for f in range(num_frames):
            assert tracks["players"][f][1]["speed"] == 0.0

    def test_velocity_clamping_caps_extreme_teleportation_jumps(self):
        estimator = SpeedDistanceEstimator(
            frame_window=5,
            frame_rate=25.0,
            enable_smoothing=True,
            max_speed_kmh=38.0,
            min_speed_mps=0.5,
            method="rolling",
        )

        tracks = {"players": []}
        # Player moving moderately, then sudden 20m jump in 1 frame (500 m/s = 1800 km/h)
        for f in range(10):
            x = 10.0 + (f * 0.2)
            if f >= 6:
                x += 20.0  # Teleportation spike
            tracks["players"].append({1: {"position_transformed": [x, 15.0]}})

        estimator.add_speed_and_distance_to_tracks(tracks)

        # Assert no frame exceeded world-class maximum sprint limit (38 km/h)
        for f in range(10):
            speed = tracks["players"][f][1]["speed"]
            assert speed <= 38.001, f"Frame {f} had supersonic speed: {speed} km/h"

    def test_streaming_estimate_chunk_stationary_deadband(self):
        estimator = SpeedDistanceEstimator(
            frame_window=5,
            frame_rate=25.0,
            enable_smoothing=True,
            max_speed_kmh=38.0,
            min_speed_mps=0.5,
        )

        tracks = {"players": []}
        for f in range(30):
            jitter = 0.3 if (f % 2 == 0) else -0.3
            tracks["players"].append(
                {1: {"position_transformed": [30.0 + jitter, 10.0]}}
            )

        estimator.estimate_chunk(tracks)

        final_dist = tracks["players"][-1][1]["distance_covered"]
        assert final_dist < 1.0, f"Streaming distance accumulated: {final_dist}m"


class TestEventBuilderTrajectoryAlignment:
    """Test suite ensuring candidate passes require actual directional ball trajectory alignment."""

    def test_pass_confirmed_when_ball_moves_toward_receiver(self):
        builder = EventBuilder(fps=25.0, min_trajectory_alignment_cosine=0.50)

        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=20,
                start_time=0.0,
                end_time=0.8,
                player_id=1,
                team_id=1,
                duration_seconds=0.8,
                frame_count=21,
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=30,
                end_frame=50,
                start_time=1.2,
                end_time=2.0,
                player_id=2,
                team_id=1,
                duration_seconds=0.8,
                frame_count=21,
            ),
        ]

        # Player 1 is at (10, 10), Player 2 is at (25, 10) (pass vector: +15, 0; speed: 37.5 m/s)
        player_tracks = []
        for _ in range(51):
            player_tracks.append(
                {
                    1: {"position_transformed": [10.0, 10.0]},
                    2: {"position_transformed": [25.0, 10.0]},
                }
            )

        # Ball travels from Player 1 at frame 20 to Player 2 at frame 30
        ball_tracks = []
        for f in range(51):
            if f < 20:
                ball_tracks.append({1: {"position_transformed": [10.0, 10.0]}})
            elif 20 <= f <= 30:
                bx = 10.0 + ((f - 20) / 10.0) * 15.0
                ball_tracks.append({1: {"position_transformed": [bx, 10.0]}})
            else:
                ball_tracks.append({1: {"position_transformed": [25.0, 10.0]}})

        events = builder.build_events(intervals, player_tracks, ball_tracks=ball_tracks)
        assert len(events) == 1
        assert events[0].event_type == EventType.CANDIDATE_PASS.value

    def test_divergent_ball_trajectory_rejected_as_candidate_pass(self):
        builder = EventBuilder(fps=25.0, min_trajectory_alignment_cosine=0.50)

        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=20,
                start_time=0.0,
                end_time=0.8,
                player_id=1,
                team_id=1,
                duration_seconds=0.8,
                frame_count=21,
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=30,
                end_frame=50,
                start_time=1.2,
                end_time=2.0,
                player_id=2,
                team_id=1,
                duration_seconds=0.8,
                frame_count=21,
            ),
        ]

        # Player 1 is at (10, 10), Player 2 is at (40, 10) (pass vector: +30, 0)
        player_tracks = [
            {
                1: {"position_transformed": [10.0, 10.0]},
                2: {"position_transformed": [40.0, 10.0]},
            }
            for _ in range(51)
        ]

        # Ball was shot towards goal / corner at (10, 50) (orthogonal vector: 0, +40)
        ball_tracks = []
        for f in range(51):
            if f < 20:
                ball_tracks.append({1: {"position_transformed": [10.0, 10.0]}})
            elif 20 <= f <= 30:
                by = 10.0 + ((f - 20) / 10.0) * 40.0
                ball_tracks.append({1: {"position_transformed": [10.0, by]}})
            else:
                ball_tracks.append({1: {"position_transformed": [10.0, 50.0]}})

        events = builder.build_events(intervals, player_tracks, ball_tracks=ball_tracks)
        assert len(events) == 1
        # Should be classified as LOOSE_BALL or UNCERTAIN_TRANSITION, NOT CANDIDATE_PASS!
        assert events[0].event_type in (
            EventType.LOOSE_BALL.value,
            EventType.UNCERTAIN_TRANSITION.value,
        )

    def test_rebound_turnaround_classified_as_loose_ball(self):
        builder = EventBuilder(fps=25.0)

        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=15,
                start_time=0.0,
                end_time=0.6,
                player_id=1,
                team_id=1,
                duration_seconds=0.6,
                frame_count=16,
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=25,
                end_frame=40,
                start_time=1.0,
                end_time=1.6,
                player_id=2,
                team_id=1,
                duration_seconds=0.6,
                frame_count=16,
            ),
        ]

        player_tracks = [
            {
                1: {"position_transformed": [10.0, 10.0]},
                2: {"position_transformed": [15.0, 10.0]},
            }
            for _ in range(41)
        ]

        # Ball moves toward post (10 -> 35), hits post, bounces back (35 -> 15)
        ball_tracks = []
        for f in range(41):
            if f < 15:
                ball_tracks.append({1: {"position_transformed": [10.0, 10.0]}})
            elif 15 <= f <= 20:
                bx = 10.0 + ((f - 15) / 5.0) * 25.0  # 10 to 35
                ball_tracks.append({1: {"position_transformed": [bx, 10.0]}})
            elif 20 < f <= 25:
                bx = 35.0 - ((f - 20) / 5.0) * 20.0  # 35 bounces back to 15
                ball_tracks.append({1: {"position_transformed": [bx, 10.0]}})
            else:
                ball_tracks.append({1: {"position_transformed": [15.0, 10.0]}})

        events = builder.build_events(intervals, player_tracks, ball_tracks=ball_tracks)
        assert len(events) == 1
        assert events[0].event_type == EventType.LOOSE_BALL.value


class TestAnnotatorOptimization:
    """Test suite ensuring annotator does not clone full frames and computes stats in O(1)."""

    def test_draw_team_ball_control_roi_blending_and_prefix_stats(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        team_control = np.array([1, 1, 1, 2, 2, 1, 2])

        # Precompute prefix sums
        t1_cumsum = np.cumsum(team_control == 1)
        t2_cumsum = np.cumsum(team_control == 2)

        annotated = FrameAnnotator.draw_team_ball_control(
            frame,
            frame_num=4,
            team_ball_control=team_control,
            t1_count=int(t1_cumsum[4]),
            t2_count=int(t2_cumsum[4]),
        )

        assert annotated.shape == (1080, 1920, 3)
        # Check that HUD region (1350 to 1900, 850 to 970) has blended pixels (> 0)
        hud_roi = annotated[850:970, 1350:1900]
        assert np.any(hud_roi > 0)
        # Rest of top-left frame should remain untouched black (0)
        assert np.all(annotated[0:100, 0:100] == 0)

    def test_draw_camera_movement_roi_blending(self):
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
        movement = [(1.5, -0.5), (2.0, 0.0), (-1.0, 1.0)]

        annotated = FrameAnnotator.draw_camera_movement(frames, movement)
        assert len(annotated) == 3
        # Check badge region (0 to 500, 0 to 100) has blended pixels
        badge_roi = annotated[0][0:100, 0:500]
        assert np.any(badge_roi > 0)
        # Region far away should remain 0
        assert np.all(annotated[0][500:600, 500:600] == 0)
