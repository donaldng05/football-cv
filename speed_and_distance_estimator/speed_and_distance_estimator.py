"""
Backward-compatibility shim. Modern code should import from football_cv.movement.
"""

from football_cv.movement import SpeedAndDistanceEstimator, SpeedDistanceEstimator

__all__ = ["SpeedAndDistanceEstimator", "SpeedDistanceEstimator"]
