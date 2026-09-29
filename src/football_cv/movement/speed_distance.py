"""
Player metric speed (km/h) and cumulative distance (m) calculation.
"""

from typing import Any

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from ..utils.geometry import measure_distance


class SpeedDistanceEstimator:
    """Estimates metric speed and total distance covered by tracked players."""

    def __init__(
        self,
        frame_window: int = 5,
        frame_rate: float = 25.0,
        minimum_displacement: float = 0.0,
        method: str = "rolling",
        enable_smoothing: bool = True,
        smoothing_factor: float = 0.25,
        max_speed_kmh: float | None = None,
        min_speed_mps: float = 0.5,
    ):
        self.frame_window = max(1, frame_window)
        self.frame_rate = frame_rate if frame_rate > 0 else 25.0
        self.minimum_displacement = minimum_displacement
        self.method = method.lower()
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        self.max_speed_kmh = max_speed_kmh
        self.min_speed_mps = min_speed_mps
        self._player_history: dict[int, list[tuple[float, float]]] = {}
        self._player_total_distance: dict[int, float] = {}
        self._player_last_pos: dict[int, tuple[float, float]] = {}
        self._player_smoothed_pos: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        """Reset internal streaming kinematic state."""
        self._player_history.clear()
        self._player_total_distance.clear()
        self._player_last_pos.clear()
        self._player_smoothed_pos.clear()

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

                    raw_pos = (float(pos[0]), float(pos[1]))
                    if self.enable_smoothing:
                        if track_id in self._player_smoothed_pos:
                            prev_s = self._player_smoothed_pos[track_id]
                            alpha = self.smoothing_factor
                            calc_pos = (
                                alpha * raw_pos[0] + (1.0 - alpha) * prev_s[0],
                                alpha * raw_pos[1] + (1.0 - alpha) * prev_s[1],
                            )
                        else:
                            calc_pos = raw_pos
                        self._player_smoothed_pos[track_id] = calc_pos
                    else:
                        calc_pos = raw_pos

                    # 1. Rolling window speed
                    if track_id not in self._player_history:
                        self._player_history[track_id] = []

                    self._player_history[track_id].append(calc_pos)
                    if len(self._player_history[track_id]) > self.frame_window:
                        self._player_history[track_id].pop(0)

                    history = self._player_history[track_id]
                    speed_mps = 0.0
                    speed_kmh = 0.0
                    if len(history) >= 2:
                        window_dist = float(measure_distance(history[0], history[-1]))
                        time_elapsed = (len(history) - 1) / self.frame_rate
                        if (
                            time_elapsed > 0
                            and window_dist
                            >= self.minimum_displacement * (len(history) - 1)
                        ):
                            raw_speed_mps = window_dist / time_elapsed
                            if (
                                self.enable_smoothing
                                and raw_speed_mps < self.min_speed_mps
                            ):
                                speed_kmh = 0.0
                            else:
                                speed_mps = raw_speed_mps
                                speed_kmh = float(speed_mps * 3.6)
                                if self.max_speed_kmh is not None:
                                    speed_kmh = min(speed_kmh, self.max_speed_kmh)
                    track_info["speed"] = speed_kmh

                    # 2. Monotonic incremental distance
                    if track_id not in self._player_total_distance:
                        self._player_total_distance[track_id] = 0.0

                    if track_id in self._player_last_pos:
                        step_dist = float(
                            measure_distance(self._player_last_pos[track_id], calc_pos)
                        )
                        step_speed = step_dist * self.frame_rate
                        # Stationary deadband check: ignore sub-walking micro jitter or stationary speed
                        is_stationary = self.enable_smoothing and (
                            (step_speed < self.min_speed_mps)
                            or (speed_kmh == 0.0 and len(history) >= 2)
                        )
                        if not is_stationary and step_dist >= self.minimum_displacement:
                            if self.max_speed_kmh is not None:
                                max_step = (self.max_speed_kmh / 3.6) / self.frame_rate
                                step_dist = min(step_dist, max_step)
                            self._player_total_distance[track_id] += step_dist

                    self._player_last_pos[track_id] = calc_pos
                    track_info["distance_covered"] = float(
                        self._player_total_distance[track_id]
                    )

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
                    if self.enable_smoothing and speed_mps < self.min_speed_mps:
                        speed_kmh = 0.0
                        dist_covered = 0.0
                    else:
                        speed_kmh = speed_mps * 3.6
                        if self.max_speed_kmh is not None:
                            speed_kmh = min(speed_kmh, self.max_speed_kmh)
                            dist_covered = min(
                                dist_covered,
                                (self.max_speed_kmh / 3.6) * time_elapsed,
                            )

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

                # Prepare positions and optionally apply smoothing filter
                smoothed_positions: dict[int, tuple[float, float]] = {}
                valid_f = []
                xs = []
                ys = []
                for f in active_frames:
                    p = object_tracks[f][track_id].get("position_transformed")
                    if p is not None:
                        valid_f.append(f)
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))

                if self.enable_smoothing and len(valid_f) >= 3:
                    n_pts = len(valid_f)
                    if savgol_filter is not None and n_pts >= 5:
                        win = min(5, n_pts if n_pts % 2 != 0 else n_pts - 1)
                        poly = min(2, win - 1)
                        smooth_xs = savgol_filter(xs, window_length=win, polyorder=poly)
                        smooth_ys = savgol_filter(ys, window_length=win, polyorder=poly)
                    else:
                        # Moving average / exponential smoothing fallback
                        smooth_xs = np.convolve(xs, np.ones(3) / 3.0, mode="same")
                        smooth_ys = np.convolve(ys, np.ones(3) / 3.0, mode="same")

                    # Detect and suppress high-frequency alternating bounding-box perspective jitter
                    diffs_x = np.diff(smooth_xs)
                    diffs_y = np.diff(smooth_ys)
                    if len(diffs_x) >= 4:
                        sign_changes_x = sum(
                            1
                            for i in range(len(diffs_x) - 1)
                            if diffs_x[i] * diffs_x[i + 1] < 0
                        )
                        sign_changes_y = sum(
                            1
                            for i in range(len(diffs_y) - 1)
                            if diffs_y[i] * diffs_y[i + 1] < 0
                        )
                        reversal_ratio = max(
                            sign_changes_x / (len(diffs_x) - 1),
                            sign_changes_y / (len(diffs_y) - 1),
                        )
                        if reversal_ratio > 0.65:
                            # Apply 3-tap binomial low-pass filter
                            smooth_xs = np.convolve(
                                smooth_xs, [0.25, 0.5, 0.25], mode="same"
                            )
                            smooth_ys = np.convolve(
                                smooth_ys, [0.25, 0.5, 0.25], mode="same"
                            )
                            smooth_xs[0] = (xs[0] + xs[1]) / 2.0
                            smooth_xs[-1] = (xs[-1] + xs[-2]) / 2.0
                            smooth_ys[0] = (ys[0] + ys[1]) / 2.0
                            smooth_ys[-1] = (ys[-1] + ys[-2]) / 2.0

                    for f, sx, sy in zip(valid_f, smooth_xs, smooth_ys, strict=False):
                        smoothed_positions[f] = (float(sx), float(sy))
                else:
                    for f, x, y in zip(valid_f, xs, ys, strict=False):
                        smoothed_positions[f] = (x, y)

                # 1. Rolling-window instantaneous velocity
                n_active = len(active_frames)
                track_speeds: dict[int, float] = {}
                for idx, f in enumerate(active_frames):
                    curr_pos = smoothed_positions.get(f)
                    if curr_pos is None:
                        track_speeds[f] = 0.0
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

                    pos_start = smoothed_positions.get(f_start)
                    pos_end = smoothed_positions.get(f_end)

                    if pos_start is None or pos_end is None or f_end == f_start:
                        track_speeds[f] = 0.0
                        object_tracks[f][track_id]["speed"] = 0.0
                        continue

                    window_dist = float(measure_distance(pos_start, pos_end))
                    time_elapsed = (f_end - f_start) / self.frame_rate

                    if time_elapsed > 0 and window_dist >= self.minimum_displacement * (
                        f_end - f_start
                    ):
                        speed_mps = window_dist / time_elapsed
                        if self.enable_smoothing and speed_mps < self.min_speed_mps:
                            speed_kmh = 0.0
                        else:
                            speed_kmh = float(speed_mps * 3.6)
                            if self.max_speed_kmh is not None:
                                speed_kmh = min(speed_kmh, self.max_speed_kmh)
                    else:
                        speed_kmh = 0.0

                    track_speeds[f] = speed_kmh
                    object_tracks[f][track_id]["speed"] = speed_kmh

                # 2. Monotonic incremental arc-length accumulation with deadband
                cum_dist = 0.0
                prev_pos = None
                prev_f = None

                for f in active_frames:
                    curr_pos = smoothed_positions.get(f)
                    if curr_pos is not None:
                        if prev_pos is not None and prev_f is not None:
                            step_dist = float(measure_distance(prev_pos, curr_pos))
                            dt = (f - prev_f) / self.frame_rate
                            step_speed = (step_dist / dt) if dt > 0 else 0.0
                            # Stationary deadband check: ignore sub-walking micro jitter or stationary window speed
                            is_stationary = self.enable_smoothing and (
                                (step_speed < self.min_speed_mps)
                                or (track_speeds.get(f, 0.0) == 0.0)
                            )
                            if (
                                not is_stationary
                                and step_dist >= self.minimum_displacement
                            ):
                                if self.max_speed_kmh is not None and dt > 0:
                                    max_step = (self.max_speed_kmh / 3.6) * dt
                                    step_dist = min(step_dist, max_step)
                                cum_dist += step_dist
                        prev_pos = curr_pos
                        prev_f = f

                    object_tracks[f][track_id]["distance_covered"] = float(cum_dist)

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
