"""
Possession and ball tracking package for football_cv.
"""

from .assigner import PlayerBallAssigner
from .interpolation import BallInterpolator

__all__ = ["BallInterpolator", "PlayerBallAssigner"]
