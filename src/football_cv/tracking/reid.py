"""
Appearance and team-consistent track sanitization to resolve ID switches and tracklet fragmentation.
"""

from typing import Any

import cv2
import numpy as np


class TrackSanitizer:
    """
    Post-processes tracker trajectories:
    - Enforces team identity invariants to reverse ID switches between opposing teams.
    - Preserves physical trajectory smoothness and prevents 1-frame teleportation spikes.
    - Stitches fragmented tracklets using spatiotemporal gating and visual appearance descriptors.
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

    @staticmethod
    def extract_appearance_features(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """
        Extract normalized spatial Lab color histogram from the upper torso of the player.
        Returns a normalized 48-dimensional float32 feature vector.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 4 or cropped.shape[1] < 4:
            return np.zeros(48, dtype=np.float32)

        # Upper torso region (jersey)
        torso = cropped[0 : max(1, cropped.shape[0] // 2), :]
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

        # 16-bin histogram per L, a, b channel -> 48-dim descriptor
        hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256])
        hist = np.concatenate([hist_l, hist_a, hist_b]).flatten().astype(np.float32)

        norm = float(np.linalg.norm(hist))
        if norm > 1e-6:
            hist /= norm
        return hist

    @staticmethod
    def compute_appearance_distance(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """
        Compute cosine distance between two normalized feature vectors in [0.0, 1.0].
        """
        if feat_a.size == 0 or feat_b.size == 0:
            return 1.0
        norm_a = float(np.linalg.norm(feat_a))
        norm_b = float(np.linalg.norm(feat_b))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 1.0
        similarity = float(np.dot(feat_a, feat_b) / (norm_a * norm_b))
        return float(np.clip(1.0 - similarity, 0.0, 1.0))

    @classmethod
    def sanitize_team_consistency(
        cls,
        player_tracks: list[dict[int, dict[str, Any]]],
        max_crossover_distance: float = 120.0,
        max_swap_velocity_px: float = 150.0,
    ) -> None:
        """
        Enforce the invariant that a physical player cannot switch teams mid-match.

        Fixes ID switches during player crossings:
        - If two opposing players intersect/collide and subsequently invert team assignments,
          performs a suffix-aware trajectory swap from collision point forward.
        - Enforces kinematic velocity gating (max_swap_velocity_px) so players NEVER teleport
          across the pitch for a single frame.
        - Fixes isolated classification noise by aligning track frame team to dominant team
          without displacing bounding boxes.
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
                    track_history.setdefault(t_id, []).append((f_idx, int(team), bbox))

        # 2. Compute the dominant/majority team for each track ID
        majority_teams: dict[int, int] = {}
        for t_id, history in track_history.items():
            teams = [item[1] for item in history]
            majority_teams[t_id] = max(set(teams), key=teams.count)

        # 3. Detect crossover events between opposing tracks
        tracks_by_dom: dict[int, list[int]] = {1: [], 2: []}
        for tid, dom in majority_teams.items():
            if dom in tracks_by_dom:
                tracks_by_dom[dom].append(tid)

        swapped_pairs: set[tuple[int, int]] = set()

        for t1 in tracks_by_dom.get(1, []):
            for t2 in tracks_by_dom.get(2, []):
                pair_key = (min(t1, t2), max(t1, t2))
                if pair_key in swapped_pairs:
                    continue

                hist1 = {h[0]: (h[1], h[2]) for h in track_history[t1]}
                hist2 = {h[0]: (h[1], h[2]) for h in track_history[t2]}
                common_frames = sorted(set(hist1.keys()) & set(hist2.keys()))
                if not common_frames:
                    continue

                # Find candidate crossover collision frame: where distance is small AND subsequent teams invert
                crossover_frame = None
                for cf in common_frames:
                    team1_cur, bbox1 = hist1[cf]
                    team2_cur, bbox2 = hist2[cf]

                    c1 = ((bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0)
                    c2 = ((bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0)
                    dist = float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

                    if (
                        dist <= max_crossover_distance
                        and team1_cur == 2
                        and team2_cur == 1
                    ):
                        crossover_frame = cf
                        break

                if crossover_frame is not None:
                    # Validate kinematic plausibility before swapping
                    prev_f = crossover_frame - 1
                    valid_kinematics = True
                    if prev_f in hist1 and prev_f in hist2:
                        prev_c1 = (
                            (hist1[prev_f][1][0] + hist1[prev_f][1][2]) / 2.0,
                            (hist1[prev_f][1][1] + hist1[prev_f][1][3]) / 2.0,
                        )
                        prev_c2 = (
                            (hist2[prev_f][1][0] + hist2[prev_f][1][2]) / 2.0,
                            (hist2[prev_f][1][1] + hist2[prev_f][1][3]) / 2.0,
                        )
                        cur_c1 = (
                            (
                                hist1[crossover_frame][1][0]
                                + hist1[crossover_frame][1][2]
                            )
                            / 2.0,
                            (
                                hist1[crossover_frame][1][1]
                                + hist1[crossover_frame][1][3]
                            )
                            / 2.0,
                        )
                        cur_c2 = (
                            (
                                hist2[crossover_frame][1][0]
                                + hist2[crossover_frame][1][2]
                            )
                            / 2.0,
                            (
                                hist2[crossover_frame][1][1]
                                + hist2[crossover_frame][1][3]
                            )
                            / 2.0,
                        )

                        # If we swap from crossover forward, track 1 gets cur_c2 and track 2 gets cur_c1
                        disp1 = float(
                            np.sqrt(
                                (cur_c2[0] - prev_c1[0]) ** 2
                                + (cur_c2[1] - prev_c1[1]) ** 2
                            )
                        )
                        disp2 = float(
                            np.sqrt(
                                (cur_c1[0] - prev_c2[0]) ** 2
                                + (cur_c1[1] - prev_c2[1]) ** 2
                            )
                        )
                        if disp1 > max_swap_velocity_px or disp2 > max_swap_velocity_px:
                            valid_kinematics = False

                    if valid_kinematics:
                        # Perform suffix swap from crossover_frame onward for all remaining co-occurring frames
                        for f in range(crossover_frame, len(player_tracks)):
                            if t1 in player_tracks[f] and t2 in player_tracks[f]:
                                p1 = player_tracks[f][t1]
                                p2 = player_tracks[f][t2]
                                p1["team"] = 1
                                p2["team"] = 2
                                player_tracks[f][t1], player_tracks[f][t2] = p2, p1
                        swapped_pairs.add(pair_key)

        # 4. Final pass: align any remaining isolated single-frame glitches to dominant team
        # without swapping bounding boxes or causing teleportation spikes
        for frame_players in player_tracks:
            for t_id, info in frame_players.items():
                dom = majority_teams.get(t_id)
                if dom is not None and info.get("team") != dom:
                    info["team"] = dom

    @classmethod
    def stitch_fragmented_tracks(
        cls,
        player_tracks: list[dict[int, dict[str, Any]]],
        max_gap_frames: int = 30,
        max_pixel_distance: float = 120.0,
        max_appearance_distance: float = 0.35,
        appearance_features: dict[int, np.ndarray] | None = None,
    ) -> None:
        """
        Stitch tracklets where Track A disappeared and Track B appeared nearby shortly after
        with matching team affiliation and compatible appearance features.
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

                    if dist > max_pixel_distance:
                        continue

                    # Check appearance similarity if feature embeddings are available
                    if (
                        appearance_features is not None
                        and tid_a in appearance_features
                        and tid_b in appearance_features
                    ):
                        app_dist = cls.compute_appearance_distance(
                            appearance_features[tid_a], appearance_features[tid_b]
                        )
                        if app_dist > max_appearance_distance:
                            continue

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
