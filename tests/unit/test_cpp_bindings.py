"""
Unit tests for C++ native extension (football_cv._core) and pybind11 bindings.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from football_cv.core import (
    get_camera_motion_estimator,
    get_perspective_transformer,
    has_cpp_core,
)

pytestmark = pytest.mark.skipif(
    not has_cpp_core(),
    reason="Native C++ extension football_cv._core is not compiled",
)

if has_cpp_core():
    import football_cv._core as native_core  # type: ignore
else:
    native_core = None


class TestPoint2DBindings:
    """Test suite for Point2D native class and operations."""

    def test_point2d_init_and_properties(self):
        p_default = native_core.Point2D()
        assert p_default.x == 0.0
        assert p_default.y == 0.0

        p = native_core.Point2D(12.5, 45.0)
        assert p.x == 12.5
        assert p.y == 45.0

        # Mutable properties
        p.x = 20.0
        p.y = 30.0
        assert p.x == 20.0
        assert p.y == 30.0

    def test_point2d_distance_and_displacement(self):
        p1 = native_core.Point2D(0.0, 0.0)
        p2 = native_core.Point2D(3.0, 4.0)

        assert math.isclose(p1.distance_to(p2), 5.0)
        assert math.isclose(p2.distance_to(p1), 5.0)

        # displacement_from: other to this (p2 to p1 = p1 - p2 = (-3, -4))
        dx, dy = p1.displacement_from(p2)
        assert dx == -3.0
        assert dy == -4.0

    def test_point2d_is_finite(self):
        p_valid = native_core.Point2D(1.0, 2.0)
        assert p_valid.is_finite()

        p_nan = native_core.Point2D(float("nan"), 2.0)
        assert not p_nan.is_finite()

        p_inf = native_core.Point2D(1.0, float("inf"))
        assert not p_inf.is_finite()

    def test_point2d_conversions_and_sequence_unpacking(self):
        p = native_core.Point2D(10.0, 25.0)

        # Sequence protocol (__len__ and __getitem__)
        assert len(p) == 2
        assert p[0] == 10.0
        assert p[1] == 25.0
        assert p[-1] == 25.0
        assert p[-2] == 10.0

        # Unpacking
        x, y = p
        assert x == 10.0
        assert y == 25.0

        # to_tuple
        assert p.to_tuple() == (10.0, 25.0)

        # to_numpy
        arr = p.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([10.0, 25.0]))

    def test_point2d_equality_and_repr(self):
        p1 = native_core.Point2D(5.0, 7.0)
        p2 = native_core.Point2D(5.0, 7.0)
        p3 = native_core.Point2D(5.0, 7.1)

        assert p1 == p2
        assert p1 != p3
        assert repr(p1) == "Point2D(x=5, y=7)"


class TestBoundingBoxBindings:
    """Test suite for BoundingBox native class and operations."""

    def test_bbox_init_and_properties(self):
        b = native_core.BoundingBox(10.0, 20.0, 50.0, 100.0)
        assert b.x1 == 10.0
        assert b.y1 == 20.0
        assert b.x2 == 50.0
        assert b.y2 == 100.0

        assert b.width() == 40.0
        assert b.height() == 80.0
        assert b.area() == 3200.0
        assert b.is_valid()
        assert b.is_finite()

    def test_bbox_center_and_foot_position(self):
        b = native_core.BoundingBox(0.0, 10.0, 20.0, 50.0)
        center = b.center()
        assert center == native_core.Point2D(10.0, 30.0)

        foot = b.foot_position()
        assert foot == native_core.Point2D(10.0, 50.0)

    def test_bbox_contains(self):
        b = native_core.BoundingBox(10.0, 10.0, 50.0, 50.0)
        assert b.contains(native_core.Point2D(25.0, 25.0))
        assert b.contains(native_core.Point2D(10.0, 10.0))  # inclusive edge
        assert not b.contains(native_core.Point2D(5.0, 25.0))
        assert not b.contains(native_core.Point2D(60.0, 25.0))

    def test_bbox_invalid_area(self):
        # x2 < x1 is invalid
        b_inv = native_core.BoundingBox(50.0, 20.0, 10.0, 100.0)
        assert not b_inv.is_valid()
        assert b_inv.area() == 0.0

    def test_bbox_sequence_unpacking_and_repr(self):
        b = native_core.BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert len(b) == 4
        assert b[0] == 1.0
        assert b[3] == 4.0

        x1, y1, x2, y2 = b
        assert (x1, y1, x2, y2) == (1.0, 2.0, 3.0, 4.0)
        assert b.to_tuple() == (1.0, 2.0, 3.0, 4.0)
        assert repr(b) == "BoundingBox(x1=1, y1=2, x2=3, y2=4)"


class TestCameraMotionBindings:
    """Test suite for CameraMotion native struct."""

    def test_camera_motion_init_and_properties(self):
        cm = native_core.CameraMotion(3.5, -2.0, False)
        assert cm.dx == 3.5
        assert cm.dy == -2.0
        assert not cm.is_scene_cut
        assert cm.to_tuple() == (3.5, -2.0)

        cm_cut = native_core.CameraMotion(0.0, 0.0, True)
        assert cm_cut.is_scene_cut
        assert repr(cm_cut) == "CameraMotion(dx=0, dy=0, is_scene_cut=True)"


class TestGeometryFunctionsBindings:
    """Test suite for standalone native geometry functions."""

    def test_geometry_helpers(self):
        bbox = native_core.BoundingBox(10.0, 20.0, 30.0, 60.0)

        center = native_core.get_center_of_bbox(bbox)
        assert center == native_core.Point2D(20.0, 40.0)

        foot = native_core.get_foot_position(bbox)
        assert foot == native_core.Point2D(20.0, 60.0)

        assert native_core.get_bbox_width(bbox) == 20.0
        assert native_core.get_bbox_height(bbox) == 40.0

        p1 = native_core.Point2D(0.0, 0.0)
        p2 = native_core.Point2D(10.0, 0.0)
        assert native_core.measure_distance(p1, p2) == 10.0
        assert native_core.measure_xy_distance(p2, p1) == (10.0, 0.0)

    def test_batch_geometry_functions(self):
        boxes = [
            native_core.BoundingBox(0.0, 0.0, 10.0, 10.0),
            native_core.BoundingBox(20.0, 20.0, 40.0, 40.0),
        ]
        centers = native_core.extract_centers(boxes)
        assert len(centers) == 2
        assert centers[0] == native_core.Point2D(5.0, 5.0)
        assert centers[1] == native_core.Point2D(30.0, 30.0)

        foots = native_core.extract_foot_positions(boxes)
        assert len(foots) == 2
        assert foots[0] == native_core.Point2D(5.0, 10.0)
        assert foots[1] == native_core.Point2D(30.0, 40.0)

    def test_filter_valid_bboxes(self):
        boxes = [
            native_core.BoundingBox(0.0, 0.0, 10.0, 10.0),  # valid
            native_core.BoundingBox(10.0, 10.0, 5.0, 5.0),  # invalid coords
            native_core.BoundingBox(float("nan"), 0.0, 10.0, 10.0),  # invalid NaN
        ]
        valid = native_core.filter_valid_bboxes(boxes)
        assert len(valid) == 1
        assert valid[0] == boxes[0]

    def test_find_nearest_point(self):
        target = native_core.Point2D(0.0, 0.0)
        candidates = [
            native_core.Point2D(10.0, 10.0),
            native_core.Point2D(2.0, 1.0),
            native_core.Point2D(5.0, 5.0),
        ]
        # Candidate 1 is nearest (~2.236 dist)
        idx = native_core.find_nearest_point(target, candidates)
        assert idx == 1

        # Beyond max_distance returns None
        idx_limited = native_core.find_nearest_point(
            target, candidates, max_distance=1.0
        )
        assert idx_limited is None


class TestPerspectiveTransformerBindings:
    """Test suite for native PerspectiveTransformer."""

    def test_default_transformer(self):
        transformer = native_core.PerspectiveTransformer()
        assert transformer.court_width == 68.0
        assert transformer.court_length == 23.32
        assert len(transformer.pixel_vertices) == 4

        # Homography matrix is 3x3 ndarray
        h = transformer.homography_matrix
        assert isinstance(h, np.ndarray)
        assert h.shape == (3, 3)

    def test_custom_vertices_formats(self):
        # 1. List of Point2D
        pts = [
            native_core.Point2D(110.0, 1035.0),
            native_core.Point2D(265.0, 275.0),
            native_core.Point2D(910.0, 260.0),
            native_core.Point2D(1640.0, 915.0),
        ]
        t1 = native_core.PerspectiveTransformer(pts, 68.0, 23.32)
        assert len(t1.pixel_vertices) == 4

        # 2. List of [x, y] lists
        pts_list = [
            [110.0, 1035.0],
            [265.0, 275.0],
            [910.0, 260.0],
            [1640.0, 915.0],
        ]
        t2 = native_core.PerspectiveTransformer(pts_list)
        assert len(t2.pixel_vertices) == 4

        # 3. NumPy array of shape (4, 2)
        arr = np.array(pts_list, dtype=np.float32)
        t3 = native_core.PerspectiveTransformer(arr)
        assert len(t3.pixel_vertices) == 4

    def test_invalid_vertex_count_raises_value_error(self):
        pts_invalid = [[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]
        with pytest.raises(ValueError, match="requires exactly 4"):
            native_core.PerspectiveTransformer(pts_invalid)

    def test_transform_point_inside_and_outside(self):
        transformer = native_core.PerspectiveTransformer()

        # Point inside field polygon
        inside_pt = native_core.Point2D(500.0, 500.0)
        assert transformer.is_point_inside(inside_pt)

        transformed = transformer.transform_point(inside_pt)
        assert transformed is not None
        assert isinstance(transformed, native_core.Point2D)
        assert 0.0 <= transformed.x <= 68.0
        assert 0.0 <= transformed.y <= 23.32

        # Transform using (x, y) overload
        transformed_xy = transformer.transform_point(500.0, 500.0)
        assert transformed_xy == transformed

        # Point outside field polygon returns None
        outside_pt = native_core.Point2D(0.0, 0.0)
        assert not transformer.is_point_inside(outside_pt)
        assert transformer.transform_point(outside_pt) is None

    def test_transform_points_batch(self):
        transformer = native_core.PerspectiveTransformer()
        pts = [
            native_core.Point2D(500.0, 500.0),  # inside
            native_core.Point2D(0.0, 0.0),  # outside
        ]
        results = transformer.transform_points(pts)
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None

    def test_homography_matches_opencv(self):
        # Compare C++ homography matrix with OpenCV cv2.getPerspectiveTransform
        src = np.array(
            [
                [110.0, 1035.0],
                [265.0, 275.0],
                [910.0, 260.0],
                [1640.0, 915.0],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                [0.0, 0.0],
                [0.0, 23.32],
                [68.0, 23.32],
                [68.0, 0.0],
            ],
            dtype=np.float32,
        )
        expected_h = cv2.getPerspectiveTransform(src, dst)

        transformer = native_core.PerspectiveTransformer(src, 68.0, 23.32)
        cpp_h = transformer.homography_matrix

        # Verify numerical match within 1e-4
        assert np.allclose(cpp_h, expected_h, atol=1e-4)

    def test_out_of_bounds_policy_enum(self):
        assert hasattr(native_core, "OutOfBoundsPolicy")
        assert native_core.OutOfBoundsPolicy.Strict is not None
        assert native_core.OutOfBoundsPolicy.Clip is not None
        assert native_core.OutOfBoundsPolicy.Extrapolate is not None

    def test_get_effective_homography_bindings(self):
        transformer = native_core.PerspectiveTransformer()
        h_base = transformer.homography_matrix
        h_eff_none = transformer.get_effective_homography()
        assert np.allclose(h_base, h_eff_none)

        # 3x3 translation camera matrix
        cam_matrix = np.array(
            [[1.0, 0.0, 10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        expected = h_base @ cam_matrix
        h_eff_cam = transformer.get_effective_homography(cam_matrix)
        assert np.allclose(h_eff_cam, expected, atol=1e-5)

    def test_transform_point_policies(self):
        transformer = native_core.PerspectiveTransformer()
        outside_pt = native_core.Point2D(0.0, 0.0)

        # Strict -> None
        assert transformer.transform_point(outside_pt, "strict") is None
        assert (
            transformer.transform_point(
                outside_pt, native_core.OutOfBoundsPolicy.Strict
            )
            is None
        )

        # Clip -> bounded within [0, court_width] x [0, court_length]
        clipped = transformer.transform_point(outside_pt, "clip")
        assert clipped is not None
        assert 0.0 <= clipped.x <= 68.0
        assert 0.0 <= clipped.y <= 23.32

        # Extrapolate -> finite values (outside standard court)
        extrapolated = transformer.transform_point(outside_pt, "extrapolate")
        assert extrapolated is not None
        assert extrapolated.is_finite()

    def test_transform_point_camera_matrix(self):
        transformer = native_core.PerspectiveTransformer()
        inside_pt = native_core.Point2D(500.0, 500.0)
        p_no_cam = transformer.transform_point(inside_pt)
        assert p_no_cam is not None

        # Camera shifted by 20 pixels
        cam_matrix = np.eye(3, dtype=np.float64)
        cam_matrix[0, 2] = 20.0
        p_cam = transformer.transform_point(
            inside_pt, camera_matrix=cam_matrix, policy="extrapolate"
        )
        assert p_cam is not None
        assert p_cam != p_no_cam

    def test_transform_points_batch_numpy(self):
        transformer = native_core.PerspectiveTransformer()
        pts = np.array([[500.0, 500.0], [0.0, 0.0], [600.0, 400.0]], dtype=np.float32)

        # 1. Strict mode (out-of-bounds become NaN)
        out_strict = transformer.transform_points_batch(
            pts, out_of_bounds_policy="strict"
        )
        assert out_strict.shape == (3, 2)
        assert out_strict.dtype == np.float32
        assert np.isfinite(out_strict[0, 0])
        assert np.isnan(out_strict[1, 0])
        assert np.isfinite(out_strict[2, 0])

        # 2. Clip mode
        out_clip = transformer.transform_points_batch(pts, out_of_bounds_policy="clip")
        assert np.all(np.isfinite(out_clip))
        assert 0.0 <= out_clip[1, 0] <= 68.0
        assert 0.0 <= out_clip[1, 1] <= 23.32

        # 3. With camera matrix
        cam_matrix = np.eye(3, dtype=np.float64)
        cam_matrix[0, 2] = 10.0
        out_cam = transformer.transform_points_batch(
            pts, camera_matrix=cam_matrix, out_of_bounds_policy="extrapolate"
        )
        assert out_cam.shape == (3, 2)
        assert np.all(np.isfinite(out_cam))

    def test_projective_horizon_singularity_rejection(self):
        transformer = native_core.PerspectiveTransformer()
        h_eff = transformer.homography_matrix
        # Calculate a point on horizon line: h31*x + h32*y + h33 = 0 -> x = -h33/h31
        horizon_x = -h_eff[2, 2] / (h_eff[2, 0] + 1e-12)
        horizon_pt = native_core.Point2D(horizon_x, 0.0)

        # Must reject with None
        assert transformer.transform_point(horizon_pt, policy="extrapolate") is None


class TestCameraMotionEstimatorBindings:
    """Test suite for native CameraMotionEstimator."""

    def test_estimator_init_and_thresholds(self):
        estimator = native_core.CameraMotionEstimator(
            minimum_distance=4.0, scene_cut_threshold=75.0
        )
        assert estimator.minimum_distance == 4.0
        assert estimator.scene_cut_threshold == 75.0

    def test_estimate_from_features(self):
        estimator = native_core.CameraMotionEstimator(5.0, 80.0)

        # 1. Below minimum_distance -> (0, 0)
        old_pts = [
            native_core.Point2D(10.0, 10.0),
            native_core.Point2D(20.0, 20.0),
        ]
        new_pts = [
            native_core.Point2D(11.0, 10.0),
            native_core.Point2D(21.0, 20.0),
        ]
        motion = estimator.estimate_from_features(old_pts, new_pts)
        assert motion.dx == 0.0
        assert motion.dy == 0.0
        assert not motion.is_scene_cut

        # 2. Significant motion (10px translation)
        new_pts_moved = [
            native_core.Point2D(20.0, 10.0),
            native_core.Point2D(30.0, 20.0),
        ]
        motion_moved = estimator.estimate_from_features(old_pts, new_pts_moved)
        # old - new = -10.0
        assert motion_moved.dx == -10.0
        assert motion_moved.dy == 0.0
        assert not motion_moved.is_scene_cut

        # 3. Scene cut (100px jump > 80.0 threshold)
        new_pts_cut = [
            native_core.Point2D(110.0, 10.0),
            native_core.Point2D(120.0, 20.0),
        ]
        motion_cut = estimator.estimate_from_features(old_pts, new_pts_cut)
        assert motion_cut.is_scene_cut
        assert motion_cut.dx == 0.0
        assert motion_cut.dy == 0.0

    def test_estimate_from_numpy_features(self):
        estimator = native_core.CameraMotionEstimator(5.0, 80.0)

        # OpenCV format: shape (N, 1, 2)
        old_arr = np.array([[[10.0, 10.0]], [[20.0, 20.0]]], dtype=np.float32)
        new_arr = np.array([[[20.0, 10.0]], [[30.0, 20.0]]], dtype=np.float32)

        motion = estimator.estimate_from_features(old_arr, new_arr)
        assert motion.dx == -10.0
        assert motion.dy == 0.0

    def test_filter_margin_points(self):
        pts = [
            native_core.Point2D(10.0, 500.0),  # left margin (0..20) -> keep
            native_core.Point2D(500.0, 500.0),  # pitch interior -> drop
            native_core.Point2D(950.0, 500.0),  # right margin (900..1050) -> keep
            native_core.Point2D(1100.0, 500.0),  # outer right -> drop
        ]
        filtered = native_core.CameraMotionEstimator.filter_margin_points(
            pts, frame_width=1920.0
        )
        assert len(filtered) == 2
        assert filtered[0].x == 10.0
        assert filtered[1].x == 950.0

    def test_accumulate_motion(self):
        steps = [
            native_core.CameraMotion(5.0, 2.0),
            native_core.CameraMotion(-2.0, 3.0),
        ]
        acc = native_core.CameraMotionEstimator.accumulate_motion(steps)
        assert len(acc) == 2
        assert acc[0] == native_core.Point2D(5.0, 2.0)
        assert acc[1] == native_core.Point2D(3.0, 5.0)

    def test_estimate_affine_partial_ransac_points(self):
        estimator = native_core.CameraMotionEstimator(5.0, 80.0)

        # Scale 1.02, angle 0.05, translation (10, -5)
        theta = 0.05
        scale = 1.02
        a = scale * math.cos(theta)
        b = scale * math.sin(theta)
        tx, ty = 10.0, -5.0

        from_pts = [
            native_core.Point2D(100.0, 100.0),
            native_core.Point2D(200.0, 150.0),
            native_core.Point2D(300.0, 250.0),
            native_core.Point2D(400.0, 100.0),
            native_core.Point2D(500.0, 300.0),
            native_core.Point2D(600.0, 200.0),
        ]
        to_pts = [
            native_core.Point2D(a * p.x - b * p.y + tx, b * p.x + a * p.y + ty)
            for p in from_pts
        ]

        res = estimator.estimate_affine_partial_ransac(from_pts, to_pts, 3.0, 100)
        assert isinstance(res, native_core.AffineResult)
        assert res.success
        assert res.inlier_count == len(from_pts)
        assert res.matrix.shape == (3, 3)
        assert np.isclose(res.matrix[0, 0], a, atol=1e-3)
        assert np.isclose(res.matrix[0, 1], -b, atol=1e-3)
        assert np.isclose(res.matrix[0, 2], tx, atol=1e-2)
        assert np.isclose(res.matrix[1, 0], b, atol=1e-3)
        assert np.isclose(res.matrix[1, 1], a, atol=1e-3)
        assert np.isclose(res.matrix[1, 2], ty, atol=1e-2)
        assert np.allclose(res.matrix[2, :], [0.0, 0.0, 1.0])

    def test_estimate_affine_partial_ransac_numpy(self):
        estimator = native_core.CameraMotionEstimator(5.0, 80.0)

        # OpenCV format: shape (N, 1, 2)
        from_arr = np.array(
            [[[100.0, 100.0]], [[200.0, 150.0]], [[300.0, 250.0]], [[400.0, 200.0]]],
            dtype=np.float32,
        )
        # Apply translation (15, -7)
        to_arr = from_arr + np.array([[[15.0, -7.0]]], dtype=np.float32)

        res = estimator.estimate_affine_partial_ransac(from_arr, to_arr, 3.0, 50)
        assert res.success
        assert res.inlier_count == 4
        assert np.isclose(res.matrix[0, 2], 15.0, atol=1e-2)
        assert np.isclose(res.matrix[1, 2], -7.0, atol=1e-2)

    def test_estimate_affine_partial_ransac_degenerate(self):
        estimator = native_core.CameraMotionEstimator(5.0, 80.0)
        # Empty array
        res_empty = estimator.estimate_affine_partial_ransac([], [])
        assert not res_empty.success
        assert res_empty.inlier_count == 0
        assert np.allclose(res_empty.matrix, np.eye(3))


class TestBackendFacadeIntegration:
    """Test suite verifying factory integration with C++ backend."""

    def test_perspective_transformer_cpp_adapter(self):
        transformer = get_perspective_transformer(backend="cpp")
        assert transformer.court_width == 68.0
        assert transformer.court_length == 23.32

        # Point inside
        point = np.array([500.0, 500.0])
        res = transformer.transform_point(point)
        assert res is not None
        assert isinstance(res, np.ndarray)
        assert res.shape == (1, 2)
        assert 0.0 <= res[0, 0] <= 68.0

        # Point outside
        assert transformer.transform_point([0.0, 0.0]) is None

        # Tracks update
        tracks = {
            "players": [{1: {"position": (500, 500), "position_adjusted": (500, 500)}}]
        }
        transformer.add_transformed_position_to_tracks(tracks)
        assert tracks["players"][0][1]["position_transformed"] is not None

    def test_camera_motion_estimator_cpp_adapter(self):
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        estimator = get_camera_motion_estimator(dummy_frame, backend="cpp")
        assert estimator.minimum_distance == 5.0
        assert estimator.scene_cut_threshold == 80.0

        # Run on simple two frames
        frames = [dummy_frame, dummy_frame]
        movement = estimator.get_camera_movement(frames)
        assert len(movement) == 2
        assert movement[0] == (0.0, 0.0)
        assert movement[1] == (0.0, 0.0)
        assert len(estimator.camera_matrices) == 2

    def test_camera_motion_estimator_cpp_adapter_native_ransac(self):
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Test explicit use_native_ransac True and False
        for use_native in [True, False]:
            estimator = get_camera_motion_estimator(
                dummy_frame, backend="cpp", use_native_ransac=use_native
            )
            assert estimator.use_native_ransac is use_native
            frames = [dummy_frame, dummy_frame]
            movement = estimator.get_camera_movement(frames)
            assert len(movement) == 2
            assert len(estimator.camera_matrices) == 2
            assert estimator.camera_matrices[0].shape == (3, 3)


class TestOnnxDetectorBindings:
    """Test suite for native OnnxDetector pybind11 bindings."""

    @pytest.fixture(autouse=True)
    def check_model(self):
        if not Path("models/best.onnx").exists():
            pytest.skip("models/best.onnx not found")

    def test_onnx_detector_initialization(self):
        detector = native_core.OnnxDetector("models/best.onnx", 2)
        assert detector.input_width == 640
        assert detector.input_height == 640
        assert detector.num_classes == 4
        assert "models/best.onnx" in detector.model_path.replace("\\", "/")

    def test_detect_scalar_confidence(self):
        detector = native_core.OnnxDetector("models/best.onnx", 2)
        frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
        dets = detector.detect(frame, confidence_threshold=0.25, nms_threshold=0.50)
        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, native_core.Detection)
            assert d.confidence >= 0.25
            assert 0 <= d.class_id < 4
            assert d.bbox.is_valid()

    def test_detect_dict_class_thresholds(self):
        detector = native_core.OnnxDetector("models/best.onnx", 2)
        frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
        thresholds = {
            "ball": 0.12,
            "player": 0.30,
            "goalkeeper": 0.30,
            "referee": 0.30,
        }
        dets = detector.detect(
            frame,
            confidence_threshold=0.20,
            nms_threshold=0.50,
            class_thresholds=thresholds,
        )
        assert isinstance(dets, list)
        for d in dets:
            expected_min = thresholds.get(
                ["ball", "goalkeeper", "player", "referee"][d.class_id], 0.20
            )
            assert d.confidence >= expected_min

    def test_detect_sequence_class_thresholds(self):
        detector = native_core.OnnxDetector("models/best.onnx", 2)
        frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
        seq_thresholds = [0.12, 0.30, 0.30, 0.30]
        dets = detector.detect(frame, 0.20, 0.50, seq_thresholds)
        assert isinstance(dets, list)
        for d in dets:
            assert d.confidence >= seq_thresholds[d.class_id]

    def test_detect_batch_dict_and_list_thresholds(self):
        detector = native_core.OnnxDetector("models/best.onnx", 2)
        frame1 = np.full((640, 640, 3), 114, dtype=np.uint8)
        frame2 = np.full((640, 640, 3), 114, dtype=np.uint8)

        # Batch with dict thresholds
        thresholds = {
            "ball": 0.10,
            "player": 0.25,
            "goalkeeper": 0.25,
            "referee": 0.25,
        }
        batch_dets = detector.detect_batch([frame1, frame2], 0.25, 0.50, thresholds)
        assert len(batch_dets) == 2
        for dets in batch_dets:
            assert isinstance(dets, list)

        # Batch with list thresholds
        batch_dets_list = detector.detect_batch(
            [frame1, frame2], 0.25, 0.50, [0.10, 0.25, 0.25, 0.25]
        )
        assert len(batch_dets_list) == 2

    def test_detect_invalid_inputs(self):
        detector = native_core.OnnxDetector("models/best.onnx", 2)
        with pytest.raises(ValueError):
            detector.detect(np.zeros((10, 10), dtype=np.uint8))  # 2D instead of 3D
        with pytest.raises(ValueError):
            detector.detect(
                np.zeros((10, 10, 4), dtype=np.uint8)
            )  # 4 channels instead of 3
        # Empty batch returns empty list gracefully
        assert detector.detect_batch([]) == []
        # Incompatible shapes in batch raise ValueError
        with pytest.raises(ValueError):
            detector.detect_batch([np.zeros((10, 10), dtype=np.uint8)])
        with pytest.raises(ValueError):
            detector.detect_batch(
                [
                    np.zeros((10, 10, 3), dtype=np.uint8),
                    np.zeros((20, 20, 3), dtype=np.uint8),
                ]
            )
