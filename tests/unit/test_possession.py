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

    def test_assign_ball_metric_space_success(self):
        assigner = PlayerBallAssigner(max_player_ball_distance_meters=2.0)
        players = {
            1: {
                "bbox": [100.0, 100.0, 140.0, 200.0],
                "position_transformed": [50.0, 30.0],  # 50m, 30m on pitch
            },
            2: {
                "bbox": [500.0, 500.0, 540.0, 600.0],
                "position_transformed": [60.0, 40.0],
            },
        }
        # Ball is at 50.8m, 30.5m (distance = sqrt(0.8^2 + 0.5^2) ≈ 0.94m <= 2.0m)
        ball_bbox = [115.0, 205.0, 125.0, 215.0]
        ball_transformed = [50.8, 30.5]

        assigned = assigner.assign_ball_to_player(
            players, ball_bbox, ball_transformed=ball_transformed
        )
        assert assigned == 1

    def test_assign_ball_metric_space_too_far(self):
        assigner = PlayerBallAssigner(max_player_ball_distance_meters=2.0)
        players = {
            1: {
                "bbox": [100.0, 100.0, 140.0, 200.0],
                "position_transformed": [50.0, 30.0],
            }
        }
        # Ball is at 53.0m, 30.0m (distance = 3.0m > 2.0m)
        ball_bbox = [100.0, 100.0, 110.0, 110.0]
        ball_transformed = [53.0, 30.0]

        assigned = assigner.assign_ball_to_player(
            players, ball_bbox, ball_transformed=ball_transformed
        )
        assert assigned == -1

    def test_assign_ball_perspective_invariance(self):
        """In perspective, background players have small pixel distances but large metric distances."""
        assigner = PlayerBallAssigner(
            max_player_ball_distance=70.0, max_player_ball_distance_meters=2.0
        )
        # Background player near far touchline: pixel distance to ball is only 40px (< 70px),
        # but in real pitch meters, 40px corresponds to 5.0m (> 2.0m threshold)
        players = {
            10: {
                "bbox": [900.0, 100.0, 920.0, 140.0],
                "position_transformed": [20.0, 95.0],
            }
        }
        ball_bbox = [930.0, 140.0, 935.0, 145.0]
        ball_transformed = [25.0, 95.0]  # 5.0m away in metric space

        # In metric space, 5.0m > 2.0m so possession must NOT be granted
        assigned = assigner.assign_ball_to_player(
            players, ball_bbox, ball_transformed=ball_transformed
        )
        assert assigned == -1

    def test_assign_ball_fallback_when_metric_none(self):
        assigner = PlayerBallAssigner(max_player_ball_distance=70.0)
        players = {
            1: {
                "bbox": [100.0, 100.0, 140.0, 200.0],
                "position_transformed": None,  # unprojected
            }
        }
        ball_bbox = [115.0, 205.0, 125.0, 215.0]
        assigned = assigner.assign_ball_to_player(
            players, ball_bbox, ball_transformed=None
        )
        assert assigned == 1


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

    def test_interpolate_15_frame_gap(self):
        # Frame 0 at (0, 0), Frame 16 at (160, 160) (gap of 15 missing frames)
        ball_frames = [{1: {"bbox": [0.0, 0.0, 10.0, 10.0]}}]
        for _ in range(15):
            ball_frames.append({})
        ball_frames.append({1: {"bbox": [160.0, 160.0, 170.0, 170.0]}})

        interpolated = BallInterpolator.interpolate_ball_positions(
            ball_frames, limit=15
        )
        assert len(interpolated) == 17
        for f in range(1, 16):
            assert 1 in interpolated[f], f"Frame {f} should be interpolated"
            expected_c = f * 10.0
            assert abs(interpolated[f][1]["bbox"][0] - expected_c) < 1e-4

    def test_interpolate_velocity_gating_rejects_impossible_teleportation(self):
        # Frame 0 at (100, 100), Frame 3 at (1900, 1900)
        # Gap = 3 - 0 = 3 frames. Distance ≈ 2545px -> ~848px/frame > 250px/frame
        ball_frames = [
            {1: {"bbox": [95.0, 95.0, 105.0, 105.0]}},
            {},
            {},
            {1: {"bbox": [1895.0, 1895.0, 1905.0, 1905.0]}},
        ]
        interpolated = BallInterpolator.interpolate_ball_positions(
            ball_frames, limit=15, max_displacement_per_frame=250.0
        )
        # Frames 1 and 2 must NOT be interpolated because of impossible jump
        assert interpolated[1] == {}
        assert interpolated[2] == {}
        assert 1 in interpolated[0]
        assert 1 in interpolated[3]

    def test_interpolate_quadratic_produces_smooth_arc(self):
        # Ball moves along a parabola y = -(x-2)^2 + 4, sampled at x = 0, 2, 4
        # At x=0: y=0; at x=2: y=4; at x=4: y=0
        ball_frames = [
            {1: {"bbox": [0.0, 0.0, 2.0, 2.0]}},  # frame 0
            {},  # frame 1 (x=1 -> y=3 in quadratic)
            {1: {"bbox": [2.0, 4.0, 4.0, 6.0]}},  # frame 2
            {},  # frame 3 (x=3 -> y=3 in quadratic)
            {1: {"bbox": [4.0, 0.0, 6.0, 2.0]}},  # frame 4
        ]
        interpolated = BallInterpolator.interpolate_ball_positions(
            ball_frames, limit=5, method="quadratic"
        )
        assert 1 in interpolated[1]
        assert 1 in interpolated[3]
        # In quadratic, at frame 1 (x=1), y1 should be close to 3.0 (rather than linear 2.0)
        assert abs(interpolated[1][1]["bbox"][1] - 3.0) < 0.2

    def test_interpolate_empty_frames(self):
        assert BallInterpolator.interpolate_ball_positions([]) == []
        assert BallInterpolator.interpolate_ball_positions([{}, {}, {}]) == [{}, {}, {}]
