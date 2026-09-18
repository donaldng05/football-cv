"""
Unit tests for speed and distance estimation.
"""

import pytest

from football_cv.movement.speed_distance import SpeedDistanceEstimator


class TestSpeedDistanceEstimator:
    def test_constant_speed_and_cumulative_distance(self, synthetic_track_sequence):
        estimator = SpeedDistanceEstimator(
            frame_window=5, frame_rate=25.0, minimum_displacement=0.0
        )
        tracks = synthetic_track_sequence

        estimator.add_speed_and_distance_to_tracks(tracks)

        # Player 1 moved 2.0 meters/frame -> 10.0 meters in 5 frames
        # Time elapsed: 5 / 25 = 0.20s
        # Speed: 10.0 / 0.20 = 50.0 m/s = 180.0 km/h
        p1_f0 = tracks["players"][0][1]
        assert "speed" in p1_f0
        assert p1_f0["speed"] == pytest.approx(180.0, rel=1e-2)
        assert p1_f0["distance_covered"] == pytest.approx(10.0, rel=1e-2)

        # In second window (frames 5-9), distance should accumulate to 20.0 meters
        p1_f5 = tracks["players"][5][1]
        assert p1_f5["distance_covered"] == pytest.approx(20.0, rel=1e-2)

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
