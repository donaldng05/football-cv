"""
Visual annotation overlays on video frames (ellipses, triangles, statistics HUD).
"""

from typing import Any

import cv2
import numpy as np

from ..utils.geometry import get_bbox_width, get_center_of_bbox, get_foot_position


class FrameAnnotator:
    """Renders visual overlays onto video frames."""

    @staticmethod
    def draw_ellipse(
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, int, int],
        track_id: int | None = None,
    ) -> np.ndarray:
        """Draw an indicator ellipse at the foot of a player/referee with an optional ID badge."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(width * 0.35)),
            angle=0,
            startAngle=-45,
            endAngle=225,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rect_width, rect_height = 40, 20
            x1_rect = x_center - rect_width // 2
            y1_rect = (y2 - rect_height // 2) + 15
            x2_rect = x_center + rect_width // 2
            y2_rect = (y2 + rect_height // 2) + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return frame

    @staticmethod
    def draw_triangle(
        frame: np.ndarray, bbox: list[float], color: tuple[int, int, int]
    ) -> np.ndarray:
        """Draw an inverted marker triangle above an entity (e.g. ball or player with ball)."""
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    @staticmethod
    def draw_team_ball_control(
        frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray
    ) -> np.ndarray:
        """Render semi-transparent possession statistics HUD."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        till_frame = team_ball_control[: frame_num + 1]
        team_1_frames = np.sum(till_frame == 1)
        team_2_frames = np.sum(till_frame == 2)
        total = team_1_frames + team_2_frames

        team_1_pct = (team_1_frames / total * 100) if total > 0 else 50.0
        team_2_pct = (team_2_frames / total * 100) if total > 0 else 50.0

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1_pct:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2_pct:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
        )
        return frame

    @staticmethod
    def draw_camera_movement(
        frames: list[np.ndarray],
        camera_movement_per_frame: list[list[float] | tuple[float, float]],
    ) -> list[np.ndarray]:
        """Render camera movement indicator badge."""
        annotated = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_mov, y_mov = camera_movement_per_frame[frame_num]
            cv2.putText(
                frame,
                f"Camera Movement: ({x_mov:.2f}, {y_mov:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            annotated.append(frame)
        return annotated

    @staticmethod
    def draw_speed_and_distance(
        frames: list[np.ndarray], tracks: dict[str, Any]
    ) -> list[np.ndarray]:
        """Render speed (km/h) and distance (m) badges below tracked players."""
        annotated = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for obj_name, object_tracks in tracks.items():
                if obj_name in ("ball", "balls", "referee", "referees"):
                    continue
                if frame_num >= len(object_tracks):
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance_covered")
                    if speed is None or distance is None:
                        continue

                    bbox = track_info["bbox"]
                    foot_pos = list(get_foot_position(bbox))
                    foot_pos[1] += 40

                    pos_int = (int(foot_pos[0]), int(foot_pos[1]))
                    cv2.putText(
                        frame,
                        f"{speed:.2f} km/h",
                        pos_int,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{distance:.2f} m",
                        (pos_int[0], pos_int[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
            annotated.append(frame)
        return annotated

    def draw_annotations(
        self,
        frames: list[np.ndarray],
        tracks: dict[str, Any],
        team_ball_control: np.ndarray,
    ) -> list[np.ndarray]:
        """Draw player ellipses, ball triangles, referee badges, and HUD onto all frames."""
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = (
                tracks["players"][frame_num]
                if frame_num < len(tracks["players"])
                else {}
            )
            ball_dict = (
                tracks["balls"][frame_num] if frame_num < len(tracks["balls"]) else {}
            )
            referee_dict = (
                tracks["referees"][frame_num]
                if frame_num < len(tracks["referees"])
                else {}
            )

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                if isinstance(color, np.ndarray):
                    color = tuple(map(int, color))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 255, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for _, ball in ball_dict.items():
                if ball.get("bbox"):
                    frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball control HUD
            if len(team_ball_control) > 0:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_frames.append(frame)

        return output_frames
