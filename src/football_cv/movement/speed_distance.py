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
        method: str = "rolling",
    ):
        self.frame_window = max(1, frame_window)
        self.frame_rate = frame_rate if frame_rate > 0 else 25.0
        self.minimum_displacement = minimum_displacement
        self.method = method.lower()
        self._player_history: dict[int, list[tuple[float, float]]] = {}
        self._player_total_distance: dict[int, float] = {}
        self._player_last_pos: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        """Reset internal streaming kinematic state."""
        self._player_history.clear()
        self._player_total_distance.clear()
        self._player_last_pos.clear()

    def estimate_chunk(self, tracks: dict[str, Any]) -> None:
        """
        Estimate speed and cumulative distance for a chunk of frames in streaming mode.
        Modifies entity dictionaries in tracks in-place.
        """
        for obj_name, object_tracks in tracks.items():
            if obj_name in ("ball", "balls", "referee", "referees"):
                continue

            for frame_track in object_tracks:
                for track_id, track_info in frame_track.items():
                    pos = track_info.get("position_transformed")
                    if pos is None:
                        track_info["speed"] = 0.0
                        track_info["distance_covered"] = float(
                            self._player_total_distance.get(track_id, 0.0)
                        )
                        continue

                    # 1. Monotonic incremental distance
                    if track_id not in self._player_total_distance:
                        self._player_total_distance[track_id] = 0.0

                    if track_id in self._player_last_pos:
                        step_dist = float(
                            measure_distance(self._player_last_pos[track_id], pos)
                        )
                        if step_dist >= self.minimum_displacement:
                            self._player_total_distance[track_id] += step_dist

                    self._player_last_pos[track_id] = pos
                    track_info["distance_covered"] = float(
                        self._player_total_distance[track_id]
                    )

                    # 2. Rolling window speed
                    if track_id not in self._player_history:
                        self._player_history[track_id] = []

                    self._player_history[track_id].append(pos)
                    if len(self._player_history[track_id]) > self.frame_window:
                        self._player_history[track_id].pop(0)

                    history = self._player_history[track_id]
                    if len(history) >= 2:
                        window_dist = float(measure_distance(history[0], history[-1]))
                        time_elapsed = (len(history) - 1) / self.frame_rate
                        if (
                            time_elapsed > 0
                            and window_dist
                            >= self.minimum_displacement * (len(history) - 1)
                        ):
                            speed_mps = window_dist / time_elapsed
                            track_info["speed"] = float(speed_mps * 3.6)
                        else:
                            track_info["speed"] = 0.0
                    else:
                        track_info["speed"] = 0.0

    def _add_speed_distance_chunk(self, tracks: dict[str, Any]) -> None:
        """Legacy 5-frame batch calculation."""
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

    def _add_speed_distance_rolling(self, tracks: dict[str, Any]) -> None:
        """Continuous rolling window instantaneous velocity and monotonic arc-length distance."""
        for obj_name, object_tracks in tracks.items():
            if obj_name in ("ball", "balls", "referee", "referees"):
                continue

            num_frames = len(object_tracks)
            if num_frames == 0:
                continue

            all_track_ids: set[int] = set()
            for frame_track in object_tracks:
                all_track_ids.update(frame_track.keys())

            half_win = self.frame_window // 2

            for track_id in sorted(all_track_ids):
                active_frames = [
                    f for f in range(num_frames) if track_id in object_tracks[f]
                ]
                if not active_frames:
                    continue

                # 1. Monotonic incremental arc-length accumulation
                cum_dist = 0.0
                prev_pos = None

                for f in active_frames:
                    curr_pos = object_tracks[f][track_id].get("position_transformed")
                    if curr_pos is not None:
                        if prev_pos is not None:
                            step_dist = float(measure_distance(prev_pos, curr_pos))
                            if step_dist >= self.minimum_displacement:
                                cum_dist += step_dist
                        prev_pos = curr_pos

                    object_tracks[f][track_id]["distance_covered"] = float(cum_dist)

                # 2. Rolling-window instantaneous velocity
                n_active = len(active_frames)
                for idx, f in enumerate(active_frames):
                    curr_pos = object_tracks[f][track_id].get("position_transformed")
                    if curr_pos is None:
                        object_tracks[f][track_id]["speed"] = 0.0
                        continue

                    idx_start = max(0, idx - half_win)
                    idx_end = min(n_active - 1, idx + half_win)

                    if (idx_end - idx_start < self.frame_window) and (
                        n_active >= self.frame_window
                    ):
                        if idx_start == 0:
                            idx_end = min(
                                n_active - 1, idx_start + self.frame_window - 1
                            )
                        elif idx_end == n_active - 1:
                            idx_start = max(0, idx_end - self.frame_window + 1)

                    f_start = active_frames[idx_start]
                    f_end = active_frames[idx_end]

                    pos_start = object_tracks[f_start][track_id].get(
                        "position_transformed"
                    )
                    pos_end = object_tracks[f_end][track_id].get("position_transformed")

                    if pos_start is None or pos_end is None or f_end == f_start:
                        object_tracks[f][track_id]["speed"] = 0.0
                        continue

                    window_dist = float(measure_distance(pos_start, pos_end))
                    time_elapsed = (f_end - f_start) / self.frame_rate

                    if time_elapsed > 0 and window_dist >= self.minimum_displacement * (
                        f_end - f_start
                    ):
                        speed_mps = window_dist / time_elapsed
                    else:
                        speed_mps = 0.0

                    object_tracks[f][track_id]["speed"] = float(speed_mps * 3.6)

    def add_speed_and_distance_to_tracks(self, tracks: dict[str, Any]) -> None:
        """
        Calculate speed (km/h) and distance (m) using transformed pitch coordinates.

        Modifies the tracks dictionary in-place.
        """
        if self.method == "chunk":
            self._add_speed_distance_chunk(tracks)
        else:
            self._add_speed_distance_rolling(tracks)

    def draw_speed_and_distance(self, frames: list, tracks: dict[str, Any]) -> list:
        """Backward-compatible wrapper delegating to FrameAnnotator."""
        from ..rendering.annotations import FrameAnnotator

        return FrameAnnotator.draw_speed_and_distance(frames, tracks)


# Alias for backward compatibility
SpeedAndDistanceEstimator = SpeedDistanceEstimator
