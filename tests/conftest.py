"""
Shared test fixtures and synthetic data generators for football_cv.
"""

from typing import Any

import numpy as np
import pytest

from football_cv.config import AppConfig, load_config


@pytest.fixture
def synthetic_video_frames() -> list[np.ndarray]:
    """Generate a synthetic 5-frame 720p RGB video sequence."""
    frames = []
    for i in range(5):
        # Create distinct color gradients per frame
        frame = np.full((720, 1280, 3), fill_value=(30 * i, 120, 50), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def synthetic_fast_config() -> AppConfig:
    """Load fast configuration profile for testing."""
    return load_config("configs/fast.yaml")


@pytest.fixture
def synthetic_track_sequence() -> dict[str, list[dict[int, dict[str, Any]]]]:
    """
    Generate a 10-frame synthetic track sequence:
      - Player 1: Moving linearly across pitch (x: 10m -> 28m, y: 15m)
      - Player 2: Stationary on pitch (x: 40m, y: 30m)
      - Player 3: Intermittent visibility (missing in frames 4 and 5)
      - Ball: Translating from Player 1 toward Player 2
      - Referee: Stationary
    """
    players: list[dict[int, dict[str, Any]]] = []
    referees: list[dict[int, dict[str, Any]]] = []
    balls: list[dict[int, dict[str, Any]]] = []

    for frame_idx in range(11):
        frame_players: dict[int, dict[str, Any]] = {}

        # Player 1 (constant velocity: 2.0 meters per frame)
        x_pitch = 10.0 + (frame_idx * 2.0)
        y_pitch = 15.0
        frame_players[1] = {
            "bbox": [100.0 + frame_idx * 10, 200.0, 140.0 + frame_idx * 10, 300.0],
            "position": (int(120.0 + frame_idx * 10), 300),
            "position_adjusted": (float(120.0 + frame_idx * 10), 300.0),
            "position_transformed": [x_pitch, y_pitch],
            "team": 1,
            "has_ball": frame_idx < 5,
        }

        # Player 2 (stationary at 40m, 30m)
        frame_players[2] = {
            "bbox": [500.0, 400.0, 540.0, 500.0],
            "position": (520, 500),
            "position_adjusted": (520.0, 500.0),
            "position_transformed": [40.0, 30.0],
            "team": 2,
            "has_ball": frame_idx >= 7,
        }

        # Player 3 (missing in frames 4 and 5)
        if frame_idx not in (4, 5):
            frame_players[3] = {
                "bbox": [800.0, 300.0, 840.0, 400.0],
                "position": (820, 400),
                "position_adjusted": (820.0, 400.0),
                "position_transformed": [55.0, 20.0],
                "team": 1,
                "has_ball": False,
            }

        players.append(frame_players)

        # Referee (stationary at 50m, 50m)
        referees.append(
            {
                100: {
                    "bbox": [600.0, 250.0, 640.0, 350.0],
                    "position": (620, 350),
                    "position_adjusted": (620.0, 350.0),
                    "position_transformed": [50.0, 50.0],
                }
            }
        )

        # Ball
        if frame_idx == 4:
            balls.append({})  # Missing detection in frame 4
        else:
            ball_x = 120.0 + (frame_idx * 40.0)
            balls.append(
                {
                    1: {
                        "bbox": [ball_x - 5, 295.0, ball_x + 5, 305.0],
                        "position": (int(ball_x), 300),
                        "position_adjusted": (float(ball_x), 300.0),
                        "position_transformed": [10.0 + frame_idx * 3.0, 15.0],
                    }
                }
            )

    return {
        "players": players,
        "referees": referees,
        "balls": balls,
    }
