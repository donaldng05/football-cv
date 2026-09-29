"""
Team classification using K-Means color clustering on player jerseys.
"""

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classifies players into Team 1 or Team 2 based on jersey color clustering."""

    def __init__(
        self,
        color_space: str = "lab",
        voting_window: int = 10,
    ):
        self.color_space = color_space.lower()
        self.voting_window = max(1, voting_window)
        self.team_colors: dict[int, np.ndarray] = {}
        self.player_team_dict: dict[int, int] = {}
        self.player_voting_history: dict[int, list[int]] = {}
        self.kmeans: KMeans | None = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """Create and fit a 2-cluster K-Means model on image pixels."""
        reshaped = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(reshaped)
        return kmeans

    def _convert_color_space(self, bgr_image: np.ndarray) -> np.ndarray:
        """Convert BGR patch to the configured color space."""
        if self.color_space == "lab":
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        if self.color_space == "hsv":
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return bgr_image

    @staticmethod
    def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
        """Convert a single Lab color vector to BGR."""
        patch = np.uint8([[np.clip(lab, 0, 255)]])
        bgr = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)[0, 0]
        return bgr.astype(float)

    @staticmethod
    def _hsv_to_bgr(hsv: np.ndarray) -> np.ndarray:
        """Convert a single HSV color vector to BGR."""
        patch = np.uint8([[np.clip(hsv, 0, 255)]])
        bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
        return bgr.astype(float)

    def get_player_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Extract dominant jersey color by isolating player upper torso from background."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        # Constrain bbox within frame dimensions
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 4 or cropped.shape[1] < 4:
            return np.array([0.0, 0.0, 0.0])

        top_half = cropped[0 : cropped.shape[0] // 2, :]
        if top_half.size == 0:
            return np.array([0.0, 0.0, 0.0])

        color_patch = self._convert_color_space(top_half)

        # High-performance native OpenCV C++ K-Means clustering (sub-millisecond execution)
        pixels = np.ascontiguousarray(color_patch.reshape(-1, 3), dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS
        )
        clustered = labels.reshape(color_patch.shape[0], color_patch.shape[1])

        # Corner pixels represent background
        corners = [
            int(clustered[0, 0]),
            int(clustered[0, -1]),
            int(clustered[-1, 0]),
            int(clustered[-1, -1]),
        ]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return centers[player_cluster].astype(float)

    def assign_team_color(
        self,
        frame_or_frames: np.ndarray | list[np.ndarray],
        player_detections_or_tracks: (
            dict[int, dict[str, Any]] | list[dict[int, dict[str, Any]]]
        ),
        sample_frames: int = 15,
        min_box_area: float = 800.0,
    ) -> None:
        """
        Determine representative colors for the two teams using single or multi-frame sampling.
        """
        player_colors: list[np.ndarray] = []

        if isinstance(frame_or_frames, list) and isinstance(
            player_detections_or_tracks, list
        ):
            # Multi-frame uniform sampling across video timeline
            n_frames = min(len(frame_or_frames), len(player_detections_or_tracks))
            if n_frames == 0:
                return

            if n_frames <= sample_frames:
                indices = list(range(n_frames))
            else:
                indices = np.linspace(
                    0, n_frames - 1, sample_frames, dtype=int
                ).tolist()

            for idx in indices:
                frame = frame_or_frames[idx]
                detections = player_detections_or_tracks[idx]
                for _, player in detections.items():
                    # Exclude goalkeepers from KMeans jersey color clustering to prevent centroid poisoning
                    if player.get("role") == "goalkeeper":
                        continue
                    bbox = player.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if bw * bh < min_box_area:
                        continue
                    color = self.get_player_color(frame, bbox)
                    if np.any(color != 0):
                        player_colors.append(color)
        else:
            # Single-frame fallback for backward compatibility
            frame = frame_or_frames  # type: ignore[assignment]
            detections = player_detections_or_tracks  # type: ignore[assignment]
            for _, player in detections.items():
                if player.get("role") == "goalkeeper":
                    continue
                bbox = player.get("bbox", [])
                if len(bbox) >= 4:
                    color = self.get_player_color(frame, bbox)
                    player_colors.append(color)

        if len(player_colors) >= 8:
            # Filter severe color outliers to protect KMeans centroids
            arr = np.array(player_colors)
            median = np.median(arr, axis=0)
            dists = np.linalg.norm(arr - median, axis=1)
            med_dist = np.median(dists)
            if med_dist > 0:
                inliers = dists <= (3.0 * med_dist)
                if np.sum(inliers) >= 4:
                    player_colors = [
                        c
                        for c, is_in in zip(player_colors, inliers, strict=False)
                        if is_in
                    ]

        if len(player_colors) < 2:
            # Fallback if fewer than 2 valid player samples are extracted
            self.team_colors[1] = np.array([255.0, 0.0, 0.0])
            self.team_colors[2] = np.array([0.0, 0.0, 255.0])
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        c1 = kmeans.cluster_centers_[0]
        c2 = kmeans.cluster_centers_[1]

        if self.color_space == "lab":
            self.team_colors[1] = self._lab_to_bgr(c1)
            self.team_colors[2] = self._lab_to_bgr(c2)
        elif self.color_space == "hsv":
            self.team_colors[1] = self._hsv_to_bgr(c1)
            self.team_colors[2] = self._hsv_to_bgr(c2)
        else:
            self.team_colors[1] = c1
            self.team_colors[2] = c2

    def get_player_team(
        self,
        frame: np.ndarray,
        player_bbox: list[float],
        player_id: int,
        role: str = "player",
        frame_players: dict[int, dict[str, Any]] | None = None,
    ) -> int:
        """
        Predict team ID (1 or 2) for a player track.
        For outfield players: uses temporal majority consensus on KMeans jersey color.
        For goalkeepers: avoids KMeans poisoning and assigns to the team defending that goal/half.
        """
        if self.kmeans is None:
            return 1

        if player_id in self.player_team_dict:
            history = self.player_voting_history.get(player_id, [])
            if len(history) >= self.voting_window:
                return self.player_team_dict[player_id]

        if role == "goalkeeper":
            # Assign goalkeeper based on pitch side relative to outfield teammates
            assigned_team = self._assign_goalkeeper_team(player_bbox, frame_players)
            if player_id not in self.player_voting_history:
                self.player_voting_history[player_id] = []
            self.player_voting_history[player_id].append(assigned_team)
            hist = self.player_voting_history[player_id]
            consensus = max(set(hist), key=hist.count)
            self.player_team_dict[player_id] = consensus
            return consensus

        player_color = self.get_player_color(frame, player_bbox)
        pred_team = int(self.kmeans.predict(player_color.reshape(1, -1))[0] + 1)

        if player_id not in self.player_voting_history:
            self.player_voting_history[player_id] = []
        self.player_voting_history[player_id].append(pred_team)

        # Majority vote among accumulated predictions so far
        hist = self.player_voting_history[player_id]
        consensus = max(set(hist), key=hist.count)
        self.player_team_dict[player_id] = consensus
        return consensus

    def _assign_goalkeeper_team(
        self,
        gk_bbox: list[float],
        frame_players: dict[int, dict[str, Any]] | None,
    ) -> int:
        """
        Assign a goalkeeper to Team 1 or Team 2 based on spatial pitch orientation
        relative to outfield players defending each side.
        """
        if not frame_players or len(gk_bbox) < 4:
            return 1

        gk_x = (gk_bbox[0] + gk_bbox[2]) / 2.0

        t1_xs = []
        t2_xs = []
        for _, p_info in frame_players.items():
            team = p_info.get("team")
            bbox = p_info.get("bbox")
            if (
                team in (1, 2)
                and bbox
                and len(bbox) >= 4
                and p_info.get("role") != "goalkeeper"
            ):
                px = (bbox[0] + bbox[2]) / 2.0
                if team == 1:
                    t1_xs.append(px)
                elif team == 2:
                    t2_xs.append(px)

        if t1_xs and t2_xs:
            mean1 = float(np.mean(t1_xs))
            mean2 = float(np.mean(t2_xs))
            midpoint = (mean1 + mean2) / 2.0
            if mean1 < mean2:
                return 1 if gk_x <= midpoint else 2
            else:
                return 2 if gk_x <= midpoint else 1

        # Fallback to closest outfield player's team
        min_dist = float("inf")
        closest_team = 1
        for _, p_info in frame_players.items():
            team = p_info.get("team")
            bbox = p_info.get("bbox")
            if team in (1, 2) and bbox and len(bbox) >= 4:
                px = (bbox[0] + bbox[2]) / 2.0
                py = (bbox[1] + bbox[3]) / 2.0
                gk_y = (gk_bbox[1] + gk_bbox[3]) / 2.0
                d = (px - gk_x) ** 2 + (py - gk_y) ** 2
                if d < min_dist:
                    min_dist = d
                    closest_team = team
        return closest_team


# Alias for backward compatibility
TeamAssigner = TeamClassifier
