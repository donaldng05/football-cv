"""
Backward-compatibility shim. Modern code should import from football_cv.utils.
"""

from football_cv.utils.geometry import (
    get_bbox_height,
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)
from football_cv.utils.video import (
    get_video_properties,
    read_video,
    save_video,
    stream_video_frames,
)

__all__ = [
    "get_bbox_height",
    "get_bbox_width",
    "get_center_of_bbox",
    "get_foot_position",
    "get_video_properties",
    "measure_distance",
    "measure_xy_distance",
    "read_video",
    "save_video",
    "stream_video_frames",
]
