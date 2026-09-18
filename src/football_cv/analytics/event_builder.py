"""
Inference engine for classifying tactical football match events
(candidate passes, turnovers, recoveries, loose balls) from possession intervals.
"""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ..possession.events import PossessionInterval


class EventType(StrEnum):
    """Categorization of inferred tactical match events."""

    CANDIDATE_PASS = "candidate_pass"
    TURNOVER = "turnover"
    RECOVERY = "recovery"
    LOOSE_BALL = "loose_ball"
    UNCERTAIN_TRANSITION = "uncertain_transition"
    EXCLUDED = "excluded"


@dataclass
class CandidateEvent:
    """Represents a discrete candidate tactical event inferred from tracking transitions."""

    event_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    from_player_id: int | None
    to_player_id: int | None
    from_team_id: int | None
    to_team_id: int | None
    start_x_pitch: float | None
    start_y_pitch: float | None
    end_x_pitch: float | None
    end_y_pitch: float | None
    start_x_image: float | None
    start_y_image: float | None
    end_x_image: float | None
    end_y_image: float | None
    transition_frames: int
    event_type: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert event dataclass to JSON-serializable dictionary."""
        return asdict(self)


class EventBuilder:
    """
    Builds candidate match events by analyzing transitions between
    consecutive possession intervals and extracting spatial coordinates.
    """

    def __init__(
        self,
        maximum_transition_frames: int = 15,
        fps: float = 25.0,
        validate_kinematics: bool = True,
        maximum_pass_speed: float = 45.0,
        minimum_track_swap_distance: float = 1.5,
    ):
        self.maximum_transition_frames = max(1, maximum_transition_frames)
        self.fps = fps if fps > 0 else 25.0
        self.validate_kinematics = validate_kinematics
        self.maximum_pass_speed = maximum_pass_speed
        self.minimum_track_swap_distance = minimum_track_swap_distance

    @staticmethod
    def _extract_coords(
        player_tracks: list[dict[int, dict[str, Any]]],
        frame_idx: int,
        player_id: int | None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Extract (pitch_x, pitch_y, image_x, image_y) for a player at a frame."""
        if player_id is None or frame_idx >= len(player_tracks):
            return None, None, None, None

        player_data = player_tracks[frame_idx].get(player_id, {})
        pitch_coords = player_data.get("position_transformed")
        img_coords = player_data.get("position_adjusted") or player_data.get("position")

        pitch_x = round(float(pitch_coords[0]), 2) if pitch_coords else None
        pitch_y = round(float(pitch_coords[1]), 2) if pitch_coords else None

        img_x = round(float(img_coords[0]), 1) if img_coords else None
        img_y = round(float(img_coords[1]), 1) if img_coords else None

        return pitch_x, pitch_y, img_x, img_y

    def build_events(
        self,
        intervals: list[PossessionInterval],
        player_tracks: list[dict[int, dict[str, Any]]],
    ) -> list[CandidateEvent]:
        """
        Infer tactical candidate events between consecutive possession intervals.

        Args:
            intervals: Chronologically ordered list of possession intervals.
            player_tracks: Frame-indexed player tracking dictionaries.

        Returns:
            List of classified CandidateEvent instances.
        """
        if not intervals:
            return []

        events: list[CandidateEvent] = []
        event_id = 1

        for idx in range(len(intervals) - 1):
            curr_int = intervals[idx]
            next_int = intervals[idx + 1]

            transition_frames = max(0, next_int.start_frame - curr_int.end_frame)
            start_frame = curr_int.end_frame
            end_frame = next_int.start_frame
            start_sec = round(start_frame / self.fps, 3)
            end_sec = round(end_frame / self.fps, 3)

            # Extract start and end positions
            sx_p, sy_p, sx_i, sy_i = self._extract_coords(
                player_tracks, start_frame, curr_int.player_id
            )
            ex_p, ey_p, ex_i, ey_i = self._extract_coords(
                player_tracks, end_frame, next_int.player_id
            )

            # Classify event type
            if curr_int.team_id == next_int.team_id:
                if curr_int.player_id == next_int.player_id:
                    # Same player regained ball after brief loose period
                    if transition_frames <= self.maximum_transition_frames:
                        event_type = EventType.RECOVERY.value
                        confidence = 0.90
                    else:
                        event_type = EventType.LOOSE_BALL.value
                        confidence = 0.75
                else:
                    # Pass candidate between teammates
                    if transition_frames <= self.maximum_transition_frames:
                        event_type = EventType.CANDIDATE_PASS.value
                        confidence = 0.85
                    else:
                        event_type = EventType.UNCERTAIN_TRANSITION.value
                        confidence = 0.50
            else:
                # Inter-team possession change
                if transition_frames <= self.maximum_transition_frames:
                    event_type = EventType.TURNOVER.value
                    confidence = 0.85
                else:
                    event_type = EventType.RECOVERY.value
                    confidence = 0.70

            # Validate kinematic plausibility for candidate passes
            if (
                self.validate_kinematics
                and event_type == EventType.CANDIDATE_PASS.value
                and sx_p is not None
                and ex_p is not None
                and sy_p is not None
                and ey_p is not None
            ):
                displacement = ((ex_p - sx_p) ** 2 + (ey_p - sy_p) ** 2) ** 0.5
                duration_sec = max(1, transition_frames) / self.fps
                speed = displacement / max(0.0001, duration_sec)

                if (
                    transition_frames <= 2
                    and displacement < self.minimum_track_swap_distance
                ):
                    # Near-zero spatial displacement across immediate frames is a tracker ID swap
                    event_type = EventType.EXCLUDED.value
                    confidence = 0.95
                elif speed > self.maximum_pass_speed:
                    # Implausible superhuman speed (> 45 m/s) indicates tracking teleportation/jump
                    event_type = EventType.UNCERTAIN_TRANSITION.value
                    confidence = 0.40

            # Downgrade if spatial coordinates are uncalibrated / off-pitch
            if sx_p is None or ex_p is None:
                if event_type != EventType.EXCLUDED.value:
                    event_type = EventType.UNCERTAIN_TRANSITION.value
                    confidence = round(confidence * 0.6, 2)

            events.append(
                CandidateEvent(
                    event_id=event_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_sec,
                    end_time=end_sec,
                    from_player_id=curr_int.player_id,
                    to_player_id=next_int.player_id,
                    from_team_id=curr_int.team_id,
                    to_team_id=next_int.team_id,
                    start_x_pitch=sx_p,
                    start_y_pitch=sy_p,
                    end_x_pitch=ex_p,
                    end_y_pitch=ey_p,
                    start_x_image=sx_i,
                    start_y_image=sy_i,
                    end_x_image=ex_i,
                    end_y_image=ey_i,
                    transition_frames=transition_frames,
                    event_type=event_type,
                    confidence=confidence,
                )
            )
            event_id += 1

        return events
