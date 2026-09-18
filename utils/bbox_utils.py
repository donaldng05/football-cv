"""
Backward-compatibility shim. Modern code should import from football_cv.utils.geometry.
"""

from football_cv.utils.geometry import (
    get_bbox_height,
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)

__all__ = [
    "get_bbox_height",
    "get_bbox_width",
    "get_center_of_bbox",
    "get_foot_position",
    "measure_distance",
    "measure_xy_distance",
]
