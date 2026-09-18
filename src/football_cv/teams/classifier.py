"""
Team classification using K-Means color clustering on player jerseys.
"""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classifies players into Team 1 or Team 2 based on jersey color clustering."""

    def __init__(self):
        self.team_colors: dict[int, np.ndarray] = {}
        self.player_team_dict: dict[int, int] = {}
        self.kmeans: KMeans | None = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """Create and fit a 2-cluster K-Means model on image pixels."""
        reshaped = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(reshaped)
        return kmeans

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

        kmeans = self.get_clustering_model(top_half)
        clustered = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])

        # Corner pixels represent background
        corners = [
            clustered[0, 0],
            clustered[0, -1],
            clustered[-1, 0],
            clustered[-1, -1],
        ]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(
        self, frame: np.ndarray, player_detections: dict[int, dict[str, Any]]
    ) -> None:
        """Determine the representative colors for the two teams using initial detections."""
        player_colors = []
        for _, player in player_detections.items():
            bbox = player["bbox"]
            color = self.get_player_color(frame, bbox)
            player_colors.append(color)

        if len(player_colors) < 2:
            # Fallback if fewer than 2 players detected
            self.team_colors[1] = np.array([255, 0, 0])
            self.team_colors[2] = np.array([0, 0, 255])
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(
        self, frame: np.ndarray, player_bbox: list[float], player_id: int
    ) -> int:
        """Predict team ID (1 or 2) for a player track."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            return 1

        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0] + 1)

        self.player_team_dict[player_id] = team_id
        return team_id


# Alias for backward compatibility
TeamAssigner = TeamClassifier
