"""
Backward-compatibility shim. Modern code should import from football_cv.teams.
"""

from football_cv.teams import TeamAssigner, TeamClassifier

__all__ = ["TeamAssigner", "TeamClassifier"]
