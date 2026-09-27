"""
Unit tests for CameraMotionEstimator (Python implementation).
"""

import numpy as np

from football_cv.camera_motion.estimator import CameraMotionEstimator


class TestCameraMotionEstimator:
    def test_filter_margin_points(self):
        points = [
            (10.0, 50.0),  # left margin
            (500.0, 300.0),  # pitch (center)
            (950.0, 400.0),  # right margin
            (1100.0, 50.0),  # outside right margin
        ]
        filtered = CameraMotionEstimator.filter_margin_points(
            points,
            frame_width=1920.0,
            left_margin_width=20.0,
            right_margin_start=900.0,
            right_margin_end=1050.0,
        )
        assert len(filtered) == 2
        assert (10.0, 50.0) in filtered
        assert (950.0, 400.0) in filtered

    def test_accumulate_motion_standard(self):
        steps = [(0.0, 0.0), (3.0, 2.0), (-1.0, 1.5), (0.0, 0.0)]
        cumulative = CameraMotionEstimator.accumulate_motion(steps)
        assert len(cumulative) == 4
        assert cumulative[0] == (0.0, 0.0)
        assert cumulative[1] == (3.0, 2.0)
        assert cumulative[2] == (2.0, 3.5)
        assert cumulative[3] == (2.0, 3.5)

    def test_accumulate_motion_scene_cut_reset(self):
        class Step:
            def __init__(self, dx, dy, is_cut):
                self.dx = dx
                self.dy = dy
                self.is_scene_cut = is_cut

        steps = [
            Step(0.0, 0.0, False),
            Step(5.0, 2.0, False),
            Step(100.0, 50.0, True),  # cut resets accumulation
            Step(2.0, 1.0, False),
        ]
        cumulative = CameraMotionEstimator.accumulate_motion(steps)
        assert len(cumulative) == 4
        assert cumulative[0] == (0.0, 0.0)
        assert cumulative[1] == (5.0, 2.0)
        assert cumulative[2] == (0.0, 0.0)
        assert cumulative[3] == (2.0, 1.0)

    def test_add_adjust_positions_to_tracks_cumulative(self):
        estimator = CameraMotionEstimator(
            first_frame=np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        tracks = {
            "players": [
                {1: {"position": (100.0, 200.0)}},
                {1: {"position": (102.0, 201.0)}},
                {1: {"position": (104.0, 202.0)}},
            ]
        }
        # Frame-to-frame displacements: (0,0), (2,1), (2,1)
        # Cumulative displacements: (0,0), (2,1), (4,2)
        displacements = [(0.0, 0.0), (2.0, 1.0), (2.0, 1.0)]

        estimator.add_adjust_positions_to_tracks(tracks, displacements, cumulative=True)

        frame0_pos = tracks["players"][0][1]["position_adjusted"]
        frame1_pos = tracks["players"][1][1]["position_adjusted"]
        frame2_pos = tracks["players"][2][1]["position_adjusted"]

        assert frame0_pos == (100.0, 200.0)
        assert frame1_pos == (100.0, 200.0)  # 102 - 2, 201 - 1
        assert frame2_pos == (100.0, 200.0)  # 104 - 4, 202 - 2

    def test_dynamic_margin_initialization(self):
        # 1920x1080 frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        est = CameraMotionEstimator(
            frame,
            margin_ratio_x=0.05,
            margin_ratio_y=0.10,
            use_dynamic_margins=True,
        )
        mask = est.features["mask"]
        # Expected left width = 1920 * 0.05 = 96
        # Expected right width = 96 (from 1920-96 = 1824 to 1920)
        # Expected top height = 1080 * 0.10 = 108
        assert mask[500, 50] == 1  # In left margin
        assert mask[500, 1850] == 1  # In right margin
        assert mask[50, 960] == 1  # In top margin
        assert mask[500, 960] == 0  # In center pitch area

    def test_legacy_margin_initialization(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        est = CameraMotionEstimator(
            frame,
            use_dynamic_margins=False,
        )
        mask = est.features["mask"]
        assert mask[500, 10] == 1  # Within 0:20
        assert mask[500, 950] == 1  # Within 900:1050
        assert mask[500, 1500] == 0

    def test_get_camera_movement_synthetic(self):
        # Create 3 synthetic textured frames simulating camera pan
        frames = []
        np.random.seed(42)
        base = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)

        # Frame 0: stationary
        frames.append(base.copy())

        # Frame 1: shift left by 10 px (camera pans right: dx = 10)
        shifted = np.zeros_like(base)
        shifted[:, :-10] = base[:, 10:]
        frames.append(shifted)

        # Frame 2: shift left by another 10 px
        shifted2 = np.zeros_like(base)
        shifted2[:, :-20] = base[:, 20:]
        frames.append(shifted2)

        est = CameraMotionEstimator(
            first_frame=frames[0],
            minimum_distance=2.0,
            use_dynamic_margins=True,
        )
        movements = est.get_camera_movement(frames)
        assert len(movements) == 3
        assert movements[0] == (0.0, 0.0)
        # The estimated displacements should be non-zero and roughly around 10
        assert movements[1][0] > 2.0
        assert movements[2][0] > 2.0
