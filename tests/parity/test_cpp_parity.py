"""
Strict numerical parity validation suite between pure-Python and native C++ vision core.

Asserts exact mathematical equivalence and validates tolerances across:
1. Geometric calculations and bounding box operations (atol=1e-9).
2. Perspective transformation and 4-point homography mapping (atol=1e-5).
3. Camera motion optical flow estimation and displacement accumulation (atol=1e-5).
4. Multi-frame real match video camera tracking.
5. End-to-end pipeline track data structures and position mutations.
"""

import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from football_cv.camera_motion.estimator import (
    CameraMotionEstimator as PyCameraMotionEstimator,
)
from football_cv.core import (
    get_camera_motion_estimator,
    get_perspective_transformer,
    has_cpp_core,
)
from football_cv.perspective.transformer import (
    PerspectiveTransformer as PyPerspectiveTransformer,
)
from football_cv.utils.geometry import (
    extract_centers as py_extract_centers,
)
from football_cv.utils.geometry import (
    extract_foot_positions as py_extract_foot_positions,
)
from football_cv.utils.geometry import (
    get_bbox_height as py_get_bbox_height,
)
from football_cv.utils.geometry import (
    get_bbox_width as py_get_bbox_width,
)
from football_cv.utils.geometry import (
    get_center_of_bbox as py_get_center_of_bbox,
)
from football_cv.utils.geometry import (
    get_foot_position as py_get_foot_position,
)
from football_cv.utils.geometry import (
    measure_distance as py_measure_distance,
)
from football_cv.utils.geometry import (
    measure_xy_distance as py_measure_xy_distance,
)

pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(
        not has_cpp_core(),
        reason="Native C++ extension football_cv._core is not compiled",
    ),
]

if has_cpp_core():
    import football_cv._core as native_core  # type: ignore
else:
    native_core = None


class TestGeometryParity:
    """Validate numerical parity for geometric primitives and bounding box operations."""

    @pytest.fixture
    def rng(self) -> np.random.RandomState:
        return np.random.RandomState(42)

    @pytest.fixture
    def random_bboxes(self, rng: np.random.RandomState) -> list[list[float]]:
        """Generate 1,000 randomized valid bounding boxes."""
        boxes = []
        for _ in range(1000):
            x1 = float(rng.uniform(0.0, 1500.0))
            y1 = float(rng.uniform(0.0, 800.0))
            w = float(rng.uniform(10.0, 300.0))
            h = float(rng.uniform(10.0, 300.0))
            boxes.append([x1, y1, x1 + w, y1 + h])
        return boxes

    def test_bbox_center_exact_parity(self, random_bboxes: list[list[float]]):
        """Center calculations must match with atol=1e-9 and preserve integer truncation contract."""
        for box in random_bboxes:
            py_center = py_get_center_of_bbox(box)
            cpp_box = native_core.BoundingBox(box[0], box[1], box[2], box[3])
            cpp_center = cpp_box.center()
            cpp_fn_center = native_core.get_center_of_bbox(cpp_box)

            # Continuous floating-point parity
            expected_cx = (box[0] + box[2]) * 0.5
            expected_cy = (box[1] + box[3]) * 0.5
            assert np.isclose(expected_cx, cpp_center.x, atol=1e-9)
            assert np.isclose(expected_cy, cpp_center.y, atol=1e-9)
            assert cpp_center == cpp_fn_center

            # Legacy Python integer contract
            assert py_center[0] == int(cpp_center.x)
            assert py_center[1] == int(cpp_center.y)

    def test_bbox_foot_position_exact_parity(self, random_bboxes: list[list[float]]):
        """Foot position calculations must match with atol=1e-9 and preserve integer truncation contract."""
        for box in random_bboxes:
            py_foot = py_get_foot_position(box)
            cpp_box = native_core.BoundingBox(box[0], box[1], box[2], box[3])
            cpp_foot = cpp_box.foot_position()
            cpp_fn_foot = native_core.get_foot_position(cpp_box)

            # Continuous floating-point parity
            expected_fx = (box[0] + box[2]) * 0.5
            expected_fy = box[3]
            assert np.isclose(expected_fx, cpp_foot.x, atol=1e-9)
            assert np.isclose(expected_fy, cpp_foot.y, atol=1e-9)
            assert cpp_foot == cpp_fn_foot

            # Legacy Python integer contract
            assert py_foot[0] == int(cpp_foot.x)
            assert py_foot[1] == int(cpp_foot.y)

    def test_bbox_dimensions_exact_parity(self, random_bboxes: list[list[float]]):
        """Width and height calculations must match exactly."""
        for box in random_bboxes:
            py_w = py_get_bbox_width(box)
            py_h = py_get_bbox_height(box)

            cpp_box = native_core.BoundingBox(box[0], box[1], box[2], box[3])
            assert np.isclose(py_w, cpp_box.width(), atol=1e-9)
            assert np.isclose(py_h, cpp_box.height(), atol=1e-9)
            assert np.isclose(py_w * py_h, cpp_box.area(), atol=1e-9)

    def test_distance_measures_exact_parity(self, rng: np.random.RandomState):
        """Euclidean and XY distance measurements must match with float64 precision."""
        for _ in range(500):
            p1 = (
                float(rng.uniform(-1000.0, 1000.0)),
                float(rng.uniform(-1000.0, 1000.0)),
            )
            p2 = (
                float(rng.uniform(-1000.0, 1000.0)),
                float(rng.uniform(-1000.0, 1000.0)),
            )

            py_dist = py_measure_distance(p1, p2)
            cpp_dist = native_core.measure_distance(
                native_core.Point2D(p1[0], p1[1]),
                native_core.Point2D(p2[0], p2[1]),
            )
            assert np.isclose(py_dist, cpp_dist, atol=1e-9)

            py_dx, py_dy = py_measure_xy_distance(p1, p2)
            cpp_dx, cpp_dy = native_core.measure_xy_distance(
                native_core.Point2D(p1[0], p1[1]),
                native_core.Point2D(p2[0], p2[1]),
            )
            assert np.isclose(py_dx, cpp_dx, atol=1e-9)
            assert np.isclose(py_dy, cpp_dy, atol=1e-9)

    def test_batch_extraction_parity(self, random_bboxes: list[list[float]]):
        """Batch extraction of centers and foot positions must match element-for-element."""
        py_centers = py_extract_centers(random_bboxes)
        cpp_boxes = [native_core.BoundingBox(*b) for b in random_bboxes]
        cpp_centers = native_core.extract_centers(cpp_boxes)

        assert len(py_centers) == len(cpp_centers)
        for (py_x, py_y), cpp_pt in zip(py_centers, cpp_centers, strict=True):
            assert py_x == int(cpp_pt.x)
            assert py_y == int(cpp_pt.y)

        py_foots = py_extract_foot_positions(random_bboxes)
        cpp_foots = native_core.extract_foot_positions(cpp_boxes)

        assert len(py_foots) == len(cpp_foots)
        for (py_x, py_y), cpp_pt in zip(py_foots, cpp_foots, strict=True):
            assert py_x == int(cpp_pt.x)
            assert py_y == int(cpp_pt.y)

    def test_bbox_edge_cases_and_validation(self):
        """Edge cases: inverted boxes, zero dimensions, NaNs, infinities."""
        # 1. Inverted boxes
        inv_box = native_core.BoundingBox(100.0, 100.0, 50.0, 50.0)
        assert not inv_box.is_valid()
        assert inv_box.area() == 0.0

        # 2. Zero-width and zero-height boxes
        zero_w = native_core.BoundingBox(50.0, 50.0, 50.0, 100.0)
        assert zero_w.is_valid()
        assert zero_w.width() == 0.0
        assert zero_w.area() == 0.0

        # 3. Non-finite values
        nan_box = native_core.BoundingBox(float("nan"), 0.0, 100.0, 100.0)
        assert not nan_box.is_valid()
        assert not nan_box.is_finite()

        inf_box = native_core.BoundingBox(0.0, 0.0, float("inf"), 100.0)
        assert not inf_box.is_valid()
        assert not inf_box.is_finite()

        # 4. filter_valid_bboxes
        mixed = [
            native_core.BoundingBox(10.0, 10.0, 50.0, 50.0),  # valid
            inv_box,  # invalid
            nan_box,  # invalid
            native_core.BoundingBox(0.0, 0.0, 100.0, 200.0),  # valid
        ]
        valid_only = native_core.filter_valid_bboxes(mixed)
        assert len(valid_only) == 2
        assert valid_only[0] == mixed[0]
        assert valid_only[1] == mixed[3]

    def test_find_nearest_point_parity(self, rng: np.random.RandomState):
        """find_nearest_point must return the mathematically closest index."""
        candidates = [
            native_core.Point2D(float(rng.uniform(0, 100)), float(rng.uniform(0, 100)))
            for _ in range(50)
        ]

        for _ in range(20):
            target = native_core.Point2D(
                float(rng.uniform(0, 100)), float(rng.uniform(0, 100))
            )

            # Compute ground truth brute force
            dists = [target.distance_to(c) for c in candidates]
            min_dist = min(dists)
            expected_idx = dists.index(min_dist)

            found_idx = native_core.find_nearest_point(target, candidates)
            assert found_idx == expected_idx

            # Threshold test
            if min_dist > 5.0:
                assert (
                    native_core.find_nearest_point(target, candidates, max_distance=5.0)
                    is None
                )


class TestPerspectiveParity:
    """Validate numerical parity for 4-point homography and metric pitch projection."""

    @pytest.fixture
    def custom_quadrilaterals(self) -> list[list[list[float]]]:
        """Set of diverse 4-point broadcast camera polygons."""
        return [
            # Standard broadcast profile
            [[110.0, 1035.0], [265.0, 275.0], [910.0, 260.0], [1640.0, 915.0]],
            # Wide angle broadcast profile
            [[50.0, 1050.0], [350.0, 200.0], [1200.0, 190.0], [1850.0, 1000.0]],
            # Asymmetrical pitch view
            [[150.0, 900.0], [200.0, 300.0], [800.0, 310.0], [1500.0, 850.0]],
            # Steep high-angle view
            [[300.0, 800.0], [400.0, 400.0], [1100.0, 400.0], [1300.0, 800.0]],
        ]

    def test_homography_matrix_numerical_parity(
        self, custom_quadrilaterals: list[list[list[float]]]
    ):
        """3x3 homography matrix must match OpenCV within atol=1e-5."""
        for vertices in custom_quadrilaterals:
            py_tf = PyPerspectiveTransformer(pixel_vertices=vertices)
            cpp_tf = native_core.PerspectiveTransformer(pixel_vertices=vertices)

            py_H = py_tf.perspective_transformer
            cpp_H = cpp_tf.homography_matrix

            np.testing.assert_allclose(
                py_H,
                cpp_H,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Homography matrix mismatch between OpenCV and C++ core",
            )

    def test_dense_grid_forward_projection_parity(self):
        """Projecting 500 coordinates inside pitch must match within sub-millimeter precision."""
        py_tf = PyPerspectiveTransformer()
        cpp_tf = native_core.PerspectiveTransformer()

        # Generate points spanning the trapezoid interior
        # In default vertices: y goes from ~260 to ~1035, x from ~265 to ~1640
        xs = np.linspace(400.0, 1000.0, 25)
        ys = np.linspace(350.0, 850.0, 20)

        for x in xs:
            for y in ys:
                pt = np.array([x, y], dtype=np.float32)

                py_transformed = py_tf.transform_point(pt)
                cpp_transformed = cpp_tf.transform_point(float(x), float(y))

                assert py_transformed is not None
                assert cpp_transformed is not None

                # Assert pitch coordinates (x_m, y_m) match within 1e-4 meters (0.1 mm)
                np.testing.assert_allclose(
                    py_transformed[0, 0],
                    cpp_transformed.x,
                    atol=1e-4,
                    err_msg=f"Discrepancy at point ({x}, {y}) for x coordinate",
                )
                np.testing.assert_allclose(
                    py_transformed[0, 1],
                    cpp_transformed.y,
                    atol=1e-4,
                    err_msg=f"Discrepancy at point ({x}, {y}) for y coordinate",
                )

    def test_boundary_and_out_of_polygon_parity(self):
        """Points outside polygon must return None identically in both implementations."""
        py_tf = PyPerspectiveTransformer()
        cpp_tf = native_core.PerspectiveTransformer()

        outside_points = [
            [0.0, 0.0],
            [1920.0, 0.0],
            [1920.0, 1080.0],
            [0.0, 1080.0],
            [-50.0, 500.0],
            [2000.0, 500.0],
            [500.0, 100.0],  # above top edge
        ]

        for pt in outside_points:
            py_res = py_tf.transform_point(pt)
            cpp_res = cpp_tf.transform_point(pt[0], pt[1])

            assert py_res is None, f"Python expected None for outside point {pt}"
            assert cpp_res is None, f"C++ expected None for outside point {pt}"

    def test_vertex_boundary_points_parity(self):
        """Corner vertices of the pitch quadrilateral must project to target boundary coordinates."""
        vertices = [
            [110.0, 1035.0],
            [265.0, 275.0],
            [910.0, 260.0],
            [1640.0, 915.0],
        ]
        py_tf = PyPerspectiveTransformer(pixel_vertices=vertices)
        cpp_tf = native_core.PerspectiveTransformer(pixel_vertices=vertices)

        for vertex in vertices:
            py_res = py_tf.transform_point(vertex)
            cpp_res = cpp_tf.transform_point(vertex[0], vertex[1])

            assert py_res is not None
            assert cpp_res is not None
            np.testing.assert_allclose(
                py_res.squeeze(),
                [cpp_res.x, cpp_res.y],
                atol=1e-3,
            )

    def test_add_transformed_position_to_tracks_parity(
        self, synthetic_track_sequence: dict[str, Any]
    ):
        """Track dictionary position_transformed mutations must match across all frames."""
        py_tf = get_perspective_transformer(backend="python")
        cpp_tf = get_perspective_transformer(backend="cpp")

        tracks_py = copy.deepcopy(synthetic_track_sequence)
        tracks_cpp = copy.deepcopy(synthetic_track_sequence)

        py_tf.add_transformed_position_to_tracks(tracks_py)
        cpp_tf.add_transformed_position_to_tracks(tracks_cpp)

        for obj_name in ["players", "referees", "ball"]:
            if obj_name not in tracks_py:
                continue
            for frame_idx in range(len(tracks_py[obj_name])):
                frame_py = tracks_py[obj_name][frame_idx]
                frame_cpp = tracks_cpp[obj_name][frame_idx]

                for track_id in frame_py:
                    pos_py = frame_py[track_id].get("position_transformed")
                    pos_cpp = frame_cpp[track_id].get("position_transformed")

                    if pos_py is None:
                        assert pos_cpp is None
                    else:
                        assert pos_cpp is not None
                        np.testing.assert_allclose(
                            pos_py,
                            pos_cpp,
                            atol=1e-4,
                            err_msg=f"Discrepancy in {obj_name} frame {frame_idx} track {track_id}",
                        )


class TestCameraMotionParity:
    """Validate numerical parity for camera motion optical flow and track compensation."""

    @pytest.fixture
    def dummy_first_frame(self) -> np.ndarray:
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_synthetic_feature_displacement_parity(self, dummy_first_frame: np.ndarray):
        """Feature displacement logic must match across stationary, pan, and scene cut cases."""
        cpp_estimator = native_core.CameraMotionEstimator(
            minimum_distance=5.0, scene_cut_threshold=80.0
        )

        old_features = [
            native_core.Point2D(100.0, 200.0),
            native_core.Point2D(300.0, 400.0),
            native_core.Point2D(500.0, 600.0),
        ]

        # 1. Zero motion
        motion_zero = cpp_estimator.estimate_from_features(old_features, old_features)
        assert motion_zero.dx == 0.0
        assert motion_zero.dy == 0.0
        assert not motion_zero.is_scene_cut

        # 2. Sub-threshold motion (dx=2, dy=1 -> dist=2.236 < 5.0)
        sub_features = [
            native_core.Point2D(102.0, 201.0),
            native_core.Point2D(302.0, 401.0),
            native_core.Point2D(502.0, 601.0),
        ]
        motion_sub = cpp_estimator.estimate_from_features(old_features, sub_features)
        assert motion_sub.dx == 0.0
        assert motion_sub.dy == 0.0
        assert not motion_sub.is_scene_cut

        # 3. Valid camera pan (dx=15, dy=-8 -> dist=17 > 5.0)
        pan_features = [
            native_core.Point2D(115.0, 192.0),
            native_core.Point2D(315.0, 392.0),
            native_core.Point2D(515.0, 592.0),
        ]
        motion_pan = cpp_estimator.estimate_from_features(old_features, pan_features)
        # Displacement is old - new = -15.0, +8.0
        assert np.isclose(motion_pan.dx, -15.0, atol=1e-9)
        assert np.isclose(motion_pan.dy, 8.0, atol=1e-9)
        assert not motion_pan.is_scene_cut

        # 4. Scene cut (dist=100 > 80.0)
        cut_features = [
            native_core.Point2D(200.0, 200.0),
            native_core.Point2D(400.0, 400.0),
            native_core.Point2D(600.0, 600.0),
        ]
        motion_cut = cpp_estimator.estimate_from_features(old_features, cut_features)
        assert motion_cut.is_scene_cut
        assert motion_cut.dx == 0.0
        assert motion_cut.dy == 0.0

    def test_real_match_video_optical_flow_parity(self):
        """
        Run 50 consecutive frames of actual match footage through both Python and C++
        estimators and verify frame-by-frame (dx, dy) translation parity.
        """
        video_path = Path("input_videos/08fd33_4.mp4")
        if not video_path.exists():
            pytest.skip("Reference match video input_videos/08fd33_4.mp4 not found")

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for _ in range(50):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        assert len(frames) >= 10, "Failed to read test frames from match video"

        py_estimator = PyCameraMotionEstimator(first_frame=frames[0])
        cpp_estimator = get_camera_motion_estimator(
            first_frame=frames[0], backend="cpp"
        )

        py_movement = py_estimator.get_camera_movement(frames)
        cpp_movement = cpp_estimator.get_camera_movement(frames)

        assert len(py_movement) == len(cpp_movement) == len(frames)

        # Displacements must match within atol=1e-5 across all frames
        np.testing.assert_allclose(
            py_movement,
            cpp_movement,
            atol=1e-5,
            err_msg="Camera motion displacement mismatch on real match footage",
        )

    def test_margin_filtering_parity(self):
        """Margin filtering must isolate left (0..20) and right (900..1050) border strips."""
        points = [
            native_core.Point2D(10.0, 300.0),  # left margin
            native_core.Point2D(50.0, 300.0),  # interior
            native_core.Point2D(950.0, 300.0),  # right margin
            native_core.Point2D(1040.0, 300.0),  # right margin
            native_core.Point2D(1060.0, 300.0),  # interior right
            native_core.Point2D(1800.0, 300.0),  # outer right
        ]

        filtered = native_core.CameraMotionEstimator.filter_margin_points(
            points, frame_width=1920.0
        )
        assert len(filtered) == 3
        assert [p.x for p in filtered] == [10.0, 950.0, 1040.0]

    def test_motion_accumulation_parity(self):
        """Accumulated displacement vectors must match cumulative sum."""
        motions = [
            native_core.CameraMotion(5.0, 1.0),
            native_core.CameraMotion(-2.0, 3.0),
            native_core.CameraMotion(4.0, -2.0),
        ]

        accumulated = native_core.CameraMotionEstimator.accumulate_motion(motions)
        assert len(accumulated) == 3
        assert accumulated[0] == native_core.Point2D(5.0, 1.0)
        assert accumulated[1] == native_core.Point2D(3.0, 4.0)
        assert accumulated[2] == native_core.Point2D(7.0, 2.0)

    def test_add_adjust_positions_to_tracks_parity(
        self, synthetic_track_sequence: dict[str, Any]
    ):
        """Track dictionary position_adjusted mutations must match between backends."""
        py_est = PyCameraMotionEstimator(
            first_frame=np.zeros((100, 100, 3), dtype=np.uint8)
        )
        cpp_est = get_camera_motion_estimator(
            first_frame=np.zeros((100, 100, 3), dtype=np.uint8), backend="cpp"
        )

        movement = [(float(i * 1.5), float(i * -0.5)) for i in range(11)]

        tracks_py = copy.deepcopy(synthetic_track_sequence)
        tracks_cpp = copy.deepcopy(synthetic_track_sequence)

        py_est.add_adjust_positions_to_tracks(tracks_py, movement)
        cpp_est.add_adjust_positions_to_tracks(tracks_cpp, movement)

        for obj_name in ["players", "referees", "ball"]:
            if obj_name not in tracks_py:
                continue
            for frame_idx in range(len(tracks_py[obj_name])):
                frame_py = tracks_py[obj_name][frame_idx]
                frame_cpp = tracks_cpp[obj_name][frame_idx]

                for track_id in frame_py:
                    pos_py = frame_py[track_id]["position_adjusted"]
                    pos_cpp = frame_cpp[track_id]["position_adjusted"]

                    np.testing.assert_allclose(
                        pos_py,
                        pos_cpp,
                        atol=1e-9,
                        err_msg=f"Discrepancy in adjusted position {obj_name} frame {frame_idx}",
                    )


class TestBackendParityIntegration:
    """Validate end-to-end integration and backend factory parity."""

    def test_factory_perspective_transformer_contract(self):
        """Factory outputs for 'python' and 'cpp' backends must satisfy identical contracts."""
        py_tf = get_perspective_transformer(backend="python")
        cpp_tf = get_perspective_transformer(backend="cpp")

        assert py_tf.court_width == cpp_tf.court_width == 68.0
        assert py_tf.court_length == cpp_tf.court_length == 23.32

        pt = np.array([600.0, 500.0], dtype=np.float32)
        py_out = py_tf.transform_point(pt)
        cpp_out = cpp_tf.transform_point(pt)

        assert py_out is not None and cpp_out is not None
        assert py_out.shape == cpp_out.shape == (1, 2)
        np.testing.assert_allclose(py_out, cpp_out, atol=1e-4)

    def test_factory_camera_motion_estimator_contract(self):
        """Factory outputs for 'python' and 'cpp' backends must satisfy identical contracts."""
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        py_est = get_camera_motion_estimator(dummy_frame, backend="python")
        cpp_est = get_camera_motion_estimator(dummy_frame, backend="cpp")

        assert py_est.minimum_distance == cpp_est.minimum_distance == 5.0
        assert py_est.scene_cut_threshold == cpp_est.scene_cut_threshold == 80.0

        frames = [dummy_frame, dummy_frame]
        py_mov = py_est.get_camera_movement(frames)
        cpp_mov = cpp_est.get_camera_movement(frames)

        assert py_mov == cpp_mov == [(0.0, 0.0), (0.0, 0.0)]
