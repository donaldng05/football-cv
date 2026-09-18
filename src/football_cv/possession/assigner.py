"""
Ball possession assignment based on player-ball spatial proximity.
"""

from typing import Any

from ..utils.geometry import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """Assigns the ball to the nearest player within a configured distance threshold."""

    def __init__(self, max_player_ball_distance: float = 70.0):
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(
        self, players: dict[int, dict[str, Any]], ball_bbox: list[float]
    ) -> int:
        """
        Identify which player currently controls the ball.

        Args:
            players: Dictionary mapping player track ID to their track information.
            ball_bbox: Bounding box [x1, y1, x2, y2] of the ball.

        Returns:
            Player track ID in possession, or -1 if no player is within threshold.
        """
        if not players or not ball_bbox:
            return -1

        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[3]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[3]), ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
