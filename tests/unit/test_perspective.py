"""
Unit tests for perspective transformation and homography mapping.
"""

import numpy as np

from football_cv.perspective.transformer import PerspectiveTransformer


class TestPerspectiveTransformer:
    def test_transformer_initialization(self):
        transformer = PerspectiveTransformer()
        assert transformer.court_width == 68.0
        assert transformer.court_length == 23.32
        assert transformer.perspective_transformer is not None

    def test_point_inside_polygon(self):
        transformer = PerspectiveTransformer()
        # Point clearly inside pixel_vertices
        point = np.array([500.0, 500.0])
        transformed = transformer.transform_point(point)
        assert transformed is not None
        assert transformed.shape == (1, 2)
        # Verify coordinates are numeric and within plausible bounds
        x_m, _ = transformed[0]
        assert 0.0 <= x_m <= 68.0

    def test_point_outside_polygon_returns_none(self):
        transformer = PerspectiveTransformer(out_of_bounds_policy="strict")
        # Point far outside field boundaries
        point = np.array([0.0, 0.0])
        transformed = transformer.transform_point(point)
        assert transformed is None

    def test_point_outside_polygon_clip_policy(self):
        transformer = PerspectiveTransformer(out_of_bounds_policy="clip")
        point = np.array([0.0, 0.0])
        transformed = transformer.transform_point(point)
        assert transformed is not None
        assert transformed.shape == (1, 2)
        x_m, y_m = transformed[0]
        assert 0.0 <= x_m <= transformer.court_width
        assert 0.0 <= y_m <= transformer.court_length

    def test_point_outside_polygon_extrapolate_policy(self):
        transformer = PerspectiveTransformer(out_of_bounds_policy="extrapolate")
        point = np.array([0.0, 0.0])
        transformed = transformer.transform_point(point)
        assert transformed is not None
        assert transformed.shape == (1, 2)

    def test_full_pitch_dimensions(self):
        transformer = PerspectiveTransformer(court_width=68.0, court_length=105.0)
        assert transformer.court_width == 68.0
        assert transformer.court_length == 105.0
