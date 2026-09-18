"""
Unit tests for ball possession assignment and ball interpolation.
"""

from football_cv.possession.assigner import PlayerBallAssigner
from football_cv.possession.interpolation import BallInterpolator


class TestPossessionAssigner:
    def test_assign_ball_to_nearest_player(self):
        assigner = PlayerBallAssigner(max_player_ball_distance=70.0)
        players = {
            1: {
                "bbox": [100.0, 100.0, 140.0, 200.0]
            },  # feet at (100, 200) and (140, 200)
            2: {"bbox": [500.0, 500.0, 540.0, 600.0]},
        }
        # Ball positioned close to player 1's foot
        ball_bbox = [115.0, 205.0, 125.0, 215.0]
        assigned = assigner.assign_ball_to_player(players, ball_bbox)
        assert assigned == 1

    def test_assign_ball_outside_max_distance_returns_minus_one(self):
        assigner = PlayerBallAssigner(max_player_ball_distance=30.0)
        players = {
            1: {"bbox": [100.0, 100.0, 140.0, 200.0]},
        }
        # Ball 100px away from player
        ball_bbox = [240.0, 200.0, 250.0, 210.0]
        assigned = assigner.assign_ball_to_player(players, ball_bbox)
        assert assigned == -1

    def test_assign_ball_empty_players(self):
        assigner = PlayerBallAssigner()
        assert assigner.assign_ball_to_player({}, [10, 10, 20, 20]) == -1


class TestBallInterpolator:
    def test_interpolate_missing_frames(self):
        # Frame 0: [0, 0, 10, 10], Frame 1: missing, Frame 2: [20, 20, 30, 30]
        ball_frames = [
            {1: {"bbox": [0.0, 0.0, 10.0, 10.0]}},
            {},
            {1: {"bbox": [20.0, 20.0, 30.0, 30.0]}},
        ]
        interpolated = BallInterpolator.interpolate_ball_positions(ball_frames)
        assert len(interpolated) == 3
        assert 1 in interpolated[1]
        mid_box = interpolated[1][1]["bbox"]
        assert mid_box == [10.0, 10.0, 20.0, 20.0]
