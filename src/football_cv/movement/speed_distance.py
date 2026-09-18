"""
Player metric speed (km/h) and cumulative distance (m) calculation.
"""

from typing import Any

from ..utils.geometry import measure_distance


class SpeedDistanceEstimator:
    """Estimates metric speed and total distance covered by tracked players."""

    def __init__(
        self,
        frame_window: int = 5,
        frame_rate: float = 25.0,
        minimum_displacement: float = 0.0,
    ):
        self.frame_window = frame_window
        self.frame_rate = frame_rate
        self.minimum_displacement = minimum_displacement

    def add_speed_and_distance_to_tracks(self, tracks: dict[str, Any]) -> None:
        """
        Calculate speed (km/h) and distance (m) using transformed pitch coordinates.

        Modifies the tracks dictionary in-place.
        """
        total_distance: dict[str, dict[int, float]] = {}

        for obj_name, object_tracks in tracks.items():
            if obj_name in ("ball", "balls", "referee", "referees"):
                continue

            num_frames = len(object_tracks)
            if num_frames == 0:
                continue

            if obj_name not in total_distance:
                total_distance[obj_name] = {}

            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)
                if last_frame <= frame_num:
                    continue

                for track_id, track_info in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = track_info.get("position_transformed")
                    end_pos = object_tracks[last_frame][track_id].get(
                        "position_transformed"
                    )

                    if start_pos is None or end_pos is None:
                        continue

                    dist_covered = measure_distance(start_pos, end_pos)
                    if dist_covered < self.minimum_displacement:
                        dist_covered = 0.0

                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_mps = (
                        (dist_covered / time_elapsed) if time_elapsed > 0 else 0.0
                    )
                    speed_kmh = speed_mps * 3.6

                    if track_id not in total_distance[obj_name]:
                        total_distance[obj_name][track_id] = 0.0
                    total_distance[obj_name][track_id] += dist_covered

                    for frame_batch in range(frame_num, last_frame):
                        if track_id in tracks[obj_name][frame_batch]:
                            tracks[obj_name][frame_batch][track_id]["speed"] = speed_kmh
                            tracks[obj_name][frame_batch][track_id][
                                "distance_covered"
                            ] = total_distance[obj_name][track_id]

    def draw_speed_and_distance(self, frames: list, tracks: dict[str, Any]) -> list:
        """Backward-compatible wrapper delegating to FrameAnnotator."""
        from ..rendering.annotations import FrameAnnotator

        return FrameAnnotator.draw_speed_and_distance(frames, tracks)


# Alias for backward compatibility
SpeedAndDistanceEstimator = SpeedDistanceEstimator
