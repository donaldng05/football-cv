"""
Unit tests for geometric and bounding box utilities.
"""

from football_cv.utils.geometry import (
    get_bbox_height,
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)


class TestGeometry:
    def test_get_center_of_bbox(self):
        bbox = [100.0, 200.0, 300.0, 400.0]
        center = get_center_of_bbox(bbox)
        assert center == (200, 300)

    def test_get_bbox_dimensions(self):
        bbox = [50.0, 80.0, 150.0, 200.0]
        assert get_bbox_width(bbox) == 100.0
        assert get_bbox_height(bbox) == 120.0

    def test_get_foot_position(self):
        bbox = [100.0, 200.0, 200.0, 400.0]
        foot = get_foot_position(bbox)
        assert foot == (150, 400)

    def test_measure_distance(self):
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        assert measure_distance(p1, p2) == 5.0

    def test_measure_xy_distance(self):
        p1 = (10.0, 20.0)
        p2 = (4.0, 8.0)
        dx, dy = measure_xy_distance(p1, p2)
        assert dx == 6.0
        assert dy == 12.0
