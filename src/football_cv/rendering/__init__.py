"""
Rendering and video annotation package for football_cv.
"""

from .annotations import FrameAnnotator
from .video_writer import AnnotatedVideoWriter

__all__ = ["AnnotatedVideoWriter", "FrameAnnotator"]
