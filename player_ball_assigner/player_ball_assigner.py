"""
Backward-compatibility shim. Modern code should import from football_cv.possession.
"""

from football_cv.possession import PlayerBallAssigner

__all__ = ["PlayerBallAssigner"]
