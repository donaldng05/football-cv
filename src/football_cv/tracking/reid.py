"""
Appearance and team-consistent track sanitization to resolve ID switches and tracklet fragmentation.
"""

from typing import Any

import numpy as np


class TrackSanitizer:
    """
    Post-processes tracker trajectories:
    - Enforces team identity invariants to reverse ID switches between opposing teams.
    - Stitches fragmented tracklets that disappear and reappear in close spatiotemporal proximity.
    """

    @staticmethod
    def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter_area = max(0.0, xb - xa) * max(0.0, yb - ya)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @classmethod
    def sanitize_team_consistency(
        cls,
        player_tracks: list[dict[int, dict[str, Any]]],
    ) -> None:
        """
        Enforce the invariant that a physical player cannot switch teams mid-match.

        If two tracks of opposing teams intersect/collide and subsequently invert team assignments,
        this method swaps their track IDs back to retain true physical identity.
        """
        if not player_tracks:
            return

        # 1. Gather all observations for each track ID: list of (frame_idx, team, bbox)
        track_history: dict[int, list[tuple[int, int, list[float]]]] = {}
        for f_idx, frame_players in enumerate(player_tracks):
            for t_id, info in frame_players.items():
                team = info.get("team")
                bbox = info.get("bbox")
                if team is not None and bbox is not None:
                    if t_id not in track_history:
                        track_history[t_id] = []
                    track_history[t_id].append((f_idx, int(team), bbox))

        # 2. Compute the dominant/majority team for each track ID
        majority_teams: dict[int, int] = {}
        for t_id, history in track_history.items():
            teams = [item[1] for item in history]
            majority_teams[t_id] = max(set(teams), key=teams.count)

        # 3. Detect and correct sudden mid-track team inversions
        swapped_pairs: set[tuple[int, int, int]] = set()
        for t_id, history in track_history.items():
            dom_team = majority_teams[t_id]
            for f_idx, current_team, _ in history:
                if current_team != dom_team:
                    corrected = False
                    # Look for an opposing player track active in this frame that inverted from the other side
                    for other_id, other_dom in majority_teams.items():
                        if other_id != t_id and other_dom == current_team:
                            pair_key = (
                                f_idx,
                                min(t_id, other_id),
                                max(t_id, other_id),
                            )
                            if pair_key in swapped_pairs:
                                corrected = True
                                break
                            other_info = player_tracks[f_idx].get(other_id)
                            if (
                                other_info is not None
                                and other_info.get("team") == dom_team
                            ):
                                # Symmetrical swap detected! Swap their track dictionary entries
                                p1 = player_tracks[f_idx][t_id]
                                p2 = player_tracks[f_idx][other_id]
                                p1["team"] = other_dom
                                p2["team"] = dom_team
                                player_tracks[f_idx][t_id] = p2
                                player_tracks[f_idx][other_id] = p1
                                swapped_pairs.add(pair_key)
                                corrected = True
                                break
                    if not corrected:
                        # Single track glitch: align with track's dominant team
                        player_tracks[f_idx][t_id]["team"] = dom_team

    @classmethod
    def stitch_fragmented_tracks(
        cls,
        player_tracks: list[dict[int, dict[str, Any]]],
        max_gap_frames: int = 30,
        max_pixel_distance: float = 120.0,
    ) -> None:
        """
        Stitch tracklets where Track A disappeared and Track B appeared nearby shortly after
        with matching team affiliation.
        """
        if not player_tracks:
            return

        # Find lifespan of each track ID: (start_frame, end_frame, last_center, team)
        lifespans: dict[int, dict[str, Any]] = {}
        for f_idx, frame_players in enumerate(player_tracks):
            for t_id, info in frame_players.items():
                bbox = info.get("bbox")
                if not bbox:
                    continue
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                team = info.get("team", 0)

                if t_id not in lifespans:
                    lifespans[t_id] = {
                        "start_frame": f_idx,
                        "end_frame": f_idx,
                        "first_center": (cx, cy),
                        "last_center": (cx, cy),
                        "team": team,
                    }
                else:
                    lifespans[t_id]["end_frame"] = f_idx
                    lifespans[t_id]["last_center"] = (cx, cy)
                    if team != 0:
                        lifespans[t_id]["team"] = team

        # Identify candidate stitches: track_a ends, track_b begins
        merge_map: dict[int, int] = {}
        sorted_tracks = sorted(
            lifespans.keys(), key=lambda tid: lifespans[tid]["start_frame"]
        )

        for i, tid_a in enumerate(sorted_tracks):
            life_a = lifespans[tid_a]
            if tid_a in merge_map:
                continue

            for tid_b in sorted_tracks[i + 1 :]:
                if tid_b in merge_map:
                    continue
                life_b = lifespans[tid_b]

                gap = life_b["start_frame"] - life_a["end_frame"]
                if 0 < gap <= max_gap_frames:
                    # Check team compatibility
                    if (
                        life_a["team"] != 0
                        and life_b["team"] != 0
                        and life_a["team"] != life_b["team"]
                    ):
                        continue

                    # Check spatial distance between end of A and start of B
                    ax, ay = life_a["last_center"]
                    bx, by = life_b["first_center"]
                    dist = float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))

                    if dist <= max_pixel_distance:
                        merge_map[tid_b] = tid_a
                        # Update life_a end point to life_b end point
                        life_a["end_frame"] = life_b["end_frame"]
                        life_a["last_center"] = life_b["last_center"]
                        break

        # Apply track re-indexing
        if merge_map:
            for frame_players in player_tracks:
                for old_id, new_id in list(merge_map.items()):
                    if old_id in frame_players and new_id not in frame_players:
                        frame_players[new_id] = frame_players.pop(old_id)
