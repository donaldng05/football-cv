"""
Ball possession assignment based on player-ball spatial proximity.
"""

from typing import Any

from ..utils.geometry import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """Assigns the ball to the nearest player within a configured distance threshold."""

    def __init__(
        self,
        max_player_ball_distance: float = 70.0,
        max_player_ball_distance_meters: float = 2.0,
        use_metric_distance: bool = True,
    ):
        self.max_player_ball_distance = max_player_ball_distance
        self.max_player_ball_distance_meters = max_player_ball_distance_meters
        self.use_metric_distance = use_metric_distance

    def assign_ball_to_player(
        self,
        players: dict[int, dict[str, Any]],
        ball_bbox: list[float],
        ball_transformed: list[float] | tuple[float, float] | None = None,
    ) -> int:
        """
        Identify which player currently controls the ball.

        Args:
            players: Dictionary mapping player track ID to their track information.
            ball_bbox: Bounding box [x1, y1, x2, y2] of the ball.
            ball_transformed: Optional metric pitch position [x_m, y_m] of the ball.

        Returns:
            Player track ID in possession, or -1 if no player is within threshold.
        """
        if not players:
            return -1

        # 1. Metric-space proximity matching if metric coordinates are available
        if self.use_metric_distance and ball_transformed is not None:
            min_metric_dist = float("inf")
            assigned_metric_player = -1
            has_metric_candidate = False

            for player_id, player in players.items():
                player_metric = player.get("position_transformed")
                if player_metric is not None and len(player_metric) >= 2:
                    has_metric_candidate = True
                    dist_m = float(measure_distance(player_metric, ball_transformed))
                    if (
                        dist_m < self.max_player_ball_distance_meters
                        and dist_m < min_metric_dist
                    ):
                        min_metric_dist = dist_m
                        assigned_metric_player = player_id

            if assigned_metric_player != -1:
                return assigned_metric_player
            if has_metric_candidate:
                # Metric coordinates were evaluated on the pitch, but no player was within physical threshold
                return -1

        # 2. Pixel-space proximity fallback
        if not ball_bbox:
            return -1

        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            if "bbox" not in player:
                continue
            player_bbox = player["bbox"]

            foot_center = ((player_bbox[0] + player_bbox[2]) / 2.0, player_bbox[3])
            distance_center = measure_distance(foot_center, ball_position)
            distance_left = measure_distance(
                (player_bbox[0], player_bbox[3]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[3]), ball_position
            )
            distance = min(distance_left, distance_right, distance_center)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
