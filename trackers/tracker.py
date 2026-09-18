"""
Backward-compatibility shim. Modern code should import from football_cv.tracking.
"""

from football_cv.tracking import ObjectTracker, Tracker

__all__ = ["ObjectTracker", "Tracker"]
