"""
Unit and integration tests for dynamic camera motion estimation and projective geometry.
"""

import numpy as np
import pytest

from football_cv.camera_motion.estimator import CameraMotionEstimator
from football_cv.perspective.transformer import PerspectiveTransformer
from football_cv.possession.assigner import PlayerBallAssigner


class TestCameraMotionGeometry:
    """Tests for affine camera motion estimation and composition."""

    def test_background_mask_excludes_entity_bboxes(self):
        frame_shape = (720, 1280, 3)
        bboxes = [[100, 100, 200, 300], [500, 400, 600, 550]]
        mask = CameraMotionEstimator.create_background_mask(
            frame_shape, bboxes, margin=10
        )

        assert mask.shape == (720, 1280)
        assert mask.dtype == np.uint8
        # Center of first bbox should be masked out (0)
        assert mask[200, 150] == 0
        # Center of second bbox should be masked out (0)
        assert mask[475, 550] == 0
        # Area outside bboxes should be 1
        assert mask[10, 10] == 1

    def test_inter_frame_matrix_pure_translation(self):
        # 10 points shifted by dx=15, dy=-5
        pts_old = np.random.uniform(100, 500, (10, 2)).astype(np.float32)
        pts_new = pts_old + np.array(
            [-15.0, 5.0], dtype=np.float32
        )  # new -> old has dx=15, dy=-5

        h = CameraMotionEstimator._estimate_inter_frame_matrix(
            pts_new, pts_old, cam_dx=15.0, cam_dy=-5.0, is_cut=False
        )

        assert h.shape == (3, 3)
        # Transform pts_new using h, should closely match pts_old
        pts_new_homo = np.hstack([pts_new, np.ones((10, 1), dtype=np.float32)])
        pts_mapped = (pts_new_homo @ h.T)[:, :2]
        np.testing.assert_allclose(pts_mapped, pts_old, atol=1e-3)

    def test_inter_frame_matrix_affine_rotation_and_translation(self):
        # Rotate by 2 degrees around center and translate
        theta = np.radians(2.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        trans = np.array([12.0, -8.0], dtype=np.float32)

        pts_old = np.random.uniform(200, 600, (20, 2)).astype(np.float32)
        # pts_old = rot_mat @ pts_new + trans  =>  pts_new = rot_mat.T @ (pts_old - trans)
        pts_new = (pts_old - trans) @ rot_mat

        h = CameraMotionEstimator._estimate_inter_frame_matrix(
            pts_new, pts_old, cam_dx=12.0, cam_dy=-8.0, is_cut=False
        )

        pts_new_homo = np.hstack([pts_new, np.ones((20, 1), dtype=np.float32)])
        pts_mapped = (pts_new_homo @ h.T)[:, :2]
        np.testing.assert_allclose(pts_mapped, pts_old, atol=1e-2)

    def test_inter_frame_matrix_scene_cut_resets_to_identity(self):
        pts_new = np.zeros((5, 2), dtype=np.float32)
        pts_old = np.zeros((5, 2), dtype=np.float32)
        h = CameraMotionEstimator._estimate_inter_frame_matrix(
            pts_new, pts_old, cam_dx=100.0, cam_dy=100.0, is_cut=True
        )
        np.testing.assert_array_equal(h, np.eye(3, dtype=np.float32))

    def test_camera_matrices_accumulation_in_chunk(self):
        first_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        estimator = CameraMotionEstimator(first_frame)

        # Chunk with 3 frames
        frames = [
            np.zeros((720, 1280, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8),
        ]
        displacements = estimator.estimate_chunk(frames)
        assert len(displacements) == 3
        assert len(estimator.last_chunk_matrices) == 3
        # Initial matrix should be 3x3 identity
        np.testing.assert_array_equal(estimator.last_chunk_matrices[0], np.eye(3))


class TestPerspectiveProjectionGeometry:
    """Tests for projective homography chaining and metric invariance."""

    @pytest.fixture
    def calibrated_transformer(self):
        # 4 broadcast pixel corners mapping to pitch 68m x 23.32m
        pixel_vertices = [
            [100.0, 1000.0],
            [300.0, 200.0],
            [900.0, 200.0],
            [1500.0, 1000.0],
        ]
        return PerspectiveTransformer(
            pixel_vertices=pixel_vertices,
            court_width=68.0,
            court_length=23.32,
            out_of_bounds_policy="strict",
        )

    def test_effective_homography_identity_camera_matrix(self, calibrated_transformer):
        h_eff = calibrated_transformer.get_effective_homography(np.eye(3))
        np.testing.assert_allclose(
            h_eff, calibrated_transformer.perspective_transformer, atol=1e-5
        )

    def test_dynamic_camera_pan_invariance(self, calibrated_transformer):
        """
        If the camera pans by (dx, dy) = (50, 0), an object fixed on the pitch
        moves in pixel coordinates from p_0 to p_1 = p_0 - (50, 0).
        With H_{1->0} translating by (+50, 0), the projected metric position
        must remain strictly invariant!
        """
        p0 = np.array([500.0, 500.0])
        metric_pos_0 = calibrated_transformer.transform_point(p0)
        assert metric_pos_0 is not None

        # Camera panned right by 50px: object appears 50px to the left in frame 1
        p1 = np.array([450.0, 500.0])
        h_1_to_0 = np.eye(3, dtype=np.float32)
        h_1_to_0[0, 2] = 50.0  # maps p1 back to p0

        metric_pos_1 = calibrated_transformer.transform_point(
            p1, camera_matrix=h_1_to_0
        )
        assert metric_pos_1 is not None

        # Transformed metric position should match frame 0 within numerical tolerance
        np.testing.assert_allclose(metric_pos_1, metric_pos_0, atol=1e-4)

    def test_projective_horizon_singularity_rejection(self, calibrated_transformer):
        # Synthesize a camera matrix that pushes a point past the projective horizon (w' <= 0)
        singular_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -100.0, 0.0]], dtype=np.float32
        )
        res = calibrated_transformer.transform_point(
            [500.0, 500.0], camera_matrix=singular_matrix
        )
        assert res is None

    def test_transform_points_batch_parity(self, calibrated_transformer):
        pts = np.array(
            [[500.0, 500.0], [600.0, 400.0], [700.0, 300.0]], dtype=np.float32
        )
        h_cam = np.eye(3, dtype=np.float32)
        h_cam[0, 2] = 20.0
        h_cam[1, 2] = -10.0

        batch_res = calibrated_transformer.transform_points_batch(
            pts, camera_matrix=h_cam
        )
        assert batch_res.shape == (3, 2)

        for i, pt in enumerate(pts):
            single_res = calibrated_transformer.transform_point(pt, camera_matrix=h_cam)
            assert single_res is not None
            np.testing.assert_allclose(batch_res[i], single_res.squeeze(), atol=1e-5)

    def test_add_transformed_position_to_tracks_with_camera_matrices(
        self, calibrated_transformer
    ):
        tracks = {
            "players": [
                {1: {"position": [500.0, 500.0]}},
                {1: {"position": [480.0, 500.0]}},
            ]
        }
        cam_mat_0 = np.eye(3, dtype=np.float32)
        cam_mat_1 = np.eye(3, dtype=np.float32)
        cam_mat_1[0, 2] = 20.0  # Camera moved 20px

        calibrated_transformer.add_transformed_position_to_tracks(
            tracks, camera_matrices=[cam_mat_0, cam_mat_1]
        )

        pos_0 = tracks["players"][0][1]["position_transformed"]
        pos_1 = tracks["players"][1][1]["position_transformed"]
        assert pos_0 is not None
        assert pos_1 is not None
        # Due to compensation, they should be identical on metric pitch
        np.testing.assert_allclose(pos_0, pos_1, atol=1e-4)


class TestPossessionFootContact:
    """Tests for player foot contact point assignment."""

    def test_ball_between_feet_assigned_via_foot_center(self):
        assigner = PlayerBallAssigner(
            max_player_ball_distance=50.0, use_metric_distance=False
        )

        # Player bbox: [x1=100, y1=100, x2=160, y2=200]
        # Bottom left: (100, 200), Bottom right: (160, 200)
        # Foot center: (130, 200)
        # If ball is at (130, 235), distance to center is 35 px (<= 50).
        # Distance to bottom-left: sqrt(30^2 + 35^2) = 46.1 px.
        # If ball is at (130, 245), distance to center is 45 px (<= 50),
        # but distance to corners is sqrt(30^2 + 45^2) = 54.1 px (> 50)!
        players = {
            10: {"bbox": [100.0, 100.0, 160.0, 200.0]},
        }
        ball_bbox = [125.0, 240.0, 135.0, 250.0]  # Center is (130, 245)

        assigned_id = assigner.assign_ball_to_player(players, ball_bbox)
        assert assigned_id == 10
