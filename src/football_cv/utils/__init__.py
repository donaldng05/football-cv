"""
Utility modules for validation, geometry, and video I/O.
"""

from .geometry import (
    get_bbox_height,
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)
from .validation import (
    run_preflight_checks,
    validate_device,
    validate_model_path,
    validate_video_path,
)
from .video import (
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
    "run_preflight_checks",
    "save_video",
    "stream_video_frames",
    "validate_device",
    "validate_model_path",
    "validate_video_path",
]
