"""
Unit tests for speed and distance estimation.
"""

import pytest

from football_cv.movement.speed_distance import SpeedDistanceEstimator


class TestSpeedDistanceEstimator:
    def test_constant_speed_and_cumulative_distance_rolling(
        self, synthetic_track_sequence
    ):
        estimator = SpeedDistanceEstimator(
            frame_window=5,
            frame_rate=25.0,
            minimum_displacement=0.0,
            method="rolling",
        )
        tracks = synthetic_track_sequence

        estimator.add_speed_and_distance_to_tracks(tracks)

        # Player 1 moved 2.0 meters/frame -> speed is 180.0 km/h
        p1_f0 = tracks["players"][0][1]
        assert "speed" in p1_f0
        assert p1_f0["speed"] == pytest.approx(180.0, rel=1e-2)

        # Under rolling monotonic accumulation, distance at frame 0 is 0.0m
        assert p1_f0["distance_covered"] == pytest.approx(0.0, rel=1e-2)

        # At frame 5 (after 5 steps of 2m), distance is 10.0m
        p1_f5 = tracks["players"][5][1]
        assert p1_f5["distance_covered"] == pytest.approx(10.0, rel=1e-2)

    def test_constant_speed_and_cumulative_distance_chunk(
        self, synthetic_track_sequence
    ):
        estimator = SpeedDistanceEstimator(
            frame_window=5,
            frame_rate=25.0,
            minimum_displacement=0.0,
            method="chunk",
        )
        tracks = synthetic_track_sequence

        estimator.add_speed_and_distance_to_tracks(tracks)

        p1_f0 = tracks["players"][0][1]
        assert p1_f0["speed"] == pytest.approx(180.0, rel=1e-2)
        # Legacy chunk broadcasts 5-frame chord backwards
        assert p1_f0["distance_covered"] == pytest.approx(10.0, rel=1e-2)

        p1_f5 = tracks["players"][5][1]
        assert p1_f5["distance_covered"] == pytest.approx(20.0, rel=1e-2)

    def test_curved_path_arc_length_accumulation(self):
        # A player running a quarter circle arc of radius R = 10m over 10 frames
        import math

        num_frames = 11
        r = 10.0
        tracks = {"players": []}

        for i in range(num_frames):
            theta = (math.pi / 2.0) * (i / (num_frames - 1))
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            tracks["players"].append(
                {1: {"position_transformed": [float(x), float(y)]}}
            )

        estimator = SpeedDistanceEstimator(method="rolling")
        estimator.add_speed_and_distance_to_tracks(tracks)

        expected_arc = (math.pi / 2.0) * r  # ≈ 15.708m
        chord = math.sqrt(r**2 + r**2)  # ≈ 14.142m

        final_dist = tracks["players"][-1][1]["distance_covered"]
        # Arc length should be much closer to true arc than straight chord
        assert final_dist > chord
        assert final_dist == pytest.approx(expected_arc, rel=0.02)

    def test_stationary_player_zero_speed(self, synthetic_track_sequence):
        estimator = SpeedDistanceEstimator(
            frame_window=5, frame_rate=25.0, minimum_displacement=0.0
        )
        tracks = synthetic_track_sequence

        estimator.add_speed_and_distance_to_tracks(tracks)

        # Player 2 position is constant at (40.0, 30.0)
        p2_f0 = tracks["players"][0][2]
        assert p2_f0["speed"] == 0.0
        assert p2_f0["distance_covered"] == 0.0

    def test_displacement_threshold_filters_minor_jitter(
        self, synthetic_track_sequence
    ):
        # Setting displacement cutoff higher than the 10.0m movement window
        estimator = SpeedDistanceEstimator(
            frame_window=5, frame_rate=25.0, minimum_displacement=15.0
        )
        tracks = synthetic_track_sequence

        estimator.add_speed_and_distance_to_tracks(tracks)

        p1_f0 = tracks["players"][0][1]
        assert p1_f0["speed"] == 0.0
        assert p1_f0["distance_covered"] == 0.0

    def test_track_dropout_handled_without_error(self, synthetic_track_sequence):
        estimator = SpeedDistanceEstimator(frame_window=5, frame_rate=25.0)
        tracks = synthetic_track_sequence

        # Player 3 is absent in frames 4 and 5
        estimator.add_speed_and_distance_to_tracks(tracks)

        assert 3 in tracks["players"][0]

    def test_empty_tracks_handled_safely(self):
        estimator = SpeedDistanceEstimator()
        empty_tracks = {"players": [], "referees": [], "balls": []}
        estimator.add_speed_and_distance_to_tracks(empty_tracks)
        assert empty_tracks["players"] == []
