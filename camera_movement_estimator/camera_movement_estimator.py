"""
Backward-compatibility shim. Modern code should import from football_cv.camera_motion.
"""

from football_cv.camera_motion import CameraMotionEstimator, CameraMovementEstimator

__all__ = ["CameraMotionEstimator", "CameraMovementEstimator"]
