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

    def __init__(
        self,
        minimum_control_frames: int = 1,
        fps: float = 25.0,
        hysteresis_frames: int = 0,
        max_gap_frames: int = 0,
    ):
        self.minimum_control_frames = max(1, minimum_control_frames)
        self.fps = fps if fps > 0 else 25.0
        self.hysteresis_frames = max(0, hysteresis_frames)
        self.max_gap_frames = max(0, max_gap_frames)

    def _smooth_frame_assignments(
        self,
        assignments: list[tuple[int, int] | None],
    ) -> list[tuple[int, int] | None]:
        """
        Apply gap-filling and hysteresis smoothing to frame-by-frame possession assignments.
        """
        n = len(assignments)
        if n == 0:
            return []

        smoothed = list(assignments)

        # 1. Short detection gap bridging: fill None gaps <= max_gap_frames between same player
        if self.max_gap_frames > 0:
            i = 0
            while i < n:
                if smoothed[i] is None:
                    gap_start = i
                    while i < n and smoothed[i] is None:
                        i += 1
                    gap_len = i - gap_start
                    if gap_len <= self.max_gap_frames and gap_start > 0 and i < n:
                        prev_val = smoothed[gap_start - 1]
                        next_val = smoothed[i]
                        if (
                            prev_val is not None
                            and next_val is not None
                            and prev_val[0] == next_val[0]
                        ):
                            for k in range(gap_start, i):
                                smoothed[k] = prev_val
                else:
                    i += 1

        # 2. Hysteresis stabilization: suppress transient runs < hysteresis_frames that revert
        if self.hysteresis_frames > 0:
            i = 0
            while i < n:
                curr = smoothed[i]
                if curr is not None:
                    run_start = i
                    while i < n and smoothed[i] == curr:
                        i += 1
                    run_len = i - run_start
                    if run_len < self.hysteresis_frames and run_start > 0 and i < n:
                        prev_val = smoothed[run_start - 1]
                        next_val = smoothed[i]
                        if (
                            prev_val is not None
                            and next_val is not None
                            and prev_val[0] == next_val[0]
                        ):
                            for k in range(run_start, i):
                                smoothed[k] = prev_val
                else:
                    i += 1

        return smoothed

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

        # Step 1: Extract raw frame-by-frame possessor: (player_id, team_id) or None
        frame_assignments: list[tuple[int, int] | None] = []
        for frame_idx, frame_players in enumerate(player_tracks):
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
                frame_assignments.append(
                    (possessor_id, possessor_team if possessor_team is not None else 1)
                )
            else:
                frame_assignments.append(None)

        # Step 2: Apply temporal smoothing if configured
        smoothed_assignments = self._smooth_frame_assignments(frame_assignments)

        # Step 3: Segment contiguous runs
        raw_runs: list[dict[str, Any]] = []
        current_run: dict[str, Any] | None = None

        for frame_idx, assignment in enumerate(smoothed_assignments):
            if assignment is not None:
                possessor_id, possessor_team = assignment
                if current_run is None:
                    # Start new possession segment
                    current_run = {
                        "player_id": possessor_id,
                        "team_id": possessor_team,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx,
                    }
                elif current_run["player_id"] == possessor_id:
                    # Continue existing segment
                    current_run["end_frame"] = frame_idx
                else:
                    # Possessor changed: close current run
                    reason = (
                        "pass"
                        if current_run["team_id"] == possessor_team
                        else "turnover"
                    )
                    current_run["termination_reason"] = reason
                    raw_runs.append(current_run)

                    current_run = {
                        "player_id": possessor_id,
                        "team_id": possessor_team,
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
