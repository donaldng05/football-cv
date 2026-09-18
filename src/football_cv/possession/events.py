"""
Domain models and extraction logic for continuous ball possession intervals.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PossessionInterval:
    """Represents a continuous segment of time during which a player controls the ball."""

    interval_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    player_id: int
    team_id: int
    duration_seconds: float
    frame_count: int
    confidence: float = 1.0
    termination_reason: str = "end_of_clip"

    def to_dict(self) -> dict[str, Any]:
        """Serialize interval attributes into a dictionary."""
        return asdict(self)


class PossessionIntervalExtractor:
    """
    Extracts and filters contiguous player ball possession intervals
    from frame-by-frame tracking detections.
    """

    def __init__(self, minimum_control_frames: int = 1, fps: float = 25.0):
        self.minimum_control_frames = max(1, minimum_control_frames)
        self.fps = fps if fps > 0 else 25.0

    def extract_intervals(
        self,
        player_tracks: list[dict[int, dict[str, Any]]],
        team_ball_control: list[int] | None = None,
    ) -> list[PossessionInterval]:
        """
        Segment frame-by-frame player tracks into continuous possession intervals.

        Args:
            player_tracks: Frame-indexed list of player track dictionaries.
            team_ball_control: Optional frame-indexed list of controlling team IDs.

        Returns:
            List of filtered PossessionInterval objects.
        """
        if not player_tracks:
            return []

        raw_runs: list[dict[str, Any]] = []
        current_run: dict[str, Any] | None = None

        for frame_idx, frame_players in enumerate(player_tracks):
            # Identify active player holding the ball in this frame
            possessor_id: int | None = None
            possessor_team: int | None = None

            for p_id, p_info in frame_players.items():
                if p_info.get("has_ball", False):
                    possessor_id = p_id
                    possessor_team = p_info.get(
                        "team",
                        team_ball_control[frame_idx]
                        if team_ball_control and frame_idx < len(team_ball_control)
                        else 1,
                    )
                    break

            if possessor_id is not None:
                if current_run is None:
                    # Start new possession segment
                    current_run = {
                        "player_id": possessor_id,
                        "team_id": possessor_team if possessor_team is not None else 1,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx,
                    }
                elif current_run["player_id"] == possessor_id:
                    # Continue existing segment
                    current_run["end_frame"] = frame_idx
                else:
                    # Possessor changed: close current run
                    next_team = possessor_team if possessor_team is not None else 1
                    reason = (
                        "pass" if current_run["team_id"] == next_team else "turnover"
                    )
                    current_run["termination_reason"] = reason
                    raw_runs.append(current_run)

                    current_run = {
                        "player_id": possessor_id,
                        "team_id": next_team,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx,
                    }
            else:
                if current_run is not None:
                    # Ball became loose
                    current_run["termination_reason"] = "lost_ball"
                    raw_runs.append(current_run)
                    current_run = None

        # Close any active trailing segment
        if current_run is not None:
            current_run["termination_reason"] = "end_of_clip"
            raw_runs.append(current_run)

        # Filter by minimum duration threshold and format as PossessionInterval
        intervals: list[PossessionInterval] = []
        interval_id = 1

        for run in raw_runs:
            frames = run["end_frame"] - run["start_frame"] + 1
            if frames >= self.minimum_control_frames:
                start_sec = round(run["start_frame"] / self.fps, 3)
                end_sec = round(run["end_frame"] / self.fps, 3)
                duration = round(frames / self.fps, 3)

                intervals.append(
                    PossessionInterval(
                        interval_id=interval_id,
                        start_frame=run["start_frame"],
                        end_frame=run["end_frame"],
                        start_time=start_sec,
                        end_time=end_sec,
                        player_id=run["player_id"],
                        team_id=run["team_id"],
                        duration_seconds=duration,
                        frame_count=frames,
                        confidence=1.0,
                        termination_reason=run.get("termination_reason", "end_of_clip"),
                    )
                )
                interval_id += 1

        return intervals
