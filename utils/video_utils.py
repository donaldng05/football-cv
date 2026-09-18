"""
Backward-compatibility shim. Modern code should import from football_cv.utils.video.
"""

from football_cv.utils.video import (
    get_video_properties,
    read_video,
    save_video,
    stream_video_frames,
)

__all__ = [
    "get_video_properties",
    "read_video",
    "save_video",
    "stream_video_frames",
]
