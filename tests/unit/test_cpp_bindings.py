"""
Unit tests for C++ native extension (football_cv._core) and pybind11 bindings.
"""

import math

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
