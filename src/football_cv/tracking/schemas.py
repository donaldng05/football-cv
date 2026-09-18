"""
Canonical tracking schemas and record representations.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrackedEntity:
    """Represents a tracked entity (player, referee, ball) in a single frame."""

    bbox: list[float]
    position: tuple[int, int] | None = None
    position_adjusted: tuple[float, float] | None = None
    position_transformed: list[float] | None = None
    speed: float | None = None
    distance_covered: float | None = None
    team: int | None = None
    team_color: tuple[int, int, int] | None = None
    has_ball: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Filter None entries if preferred for backward compatibility
        return {
            k: v for k, v in data.items() if v is not None or k in ("has_ball", "bbox")
        }


@dataclass
class MatchTracks:
    """Container for match tracking data across all frames."""

    players: list[dict[int, dict[str, Any]]] = field(default_factory=list)
    referees: list[dict[int, dict[str, Any]]] = field(default_factory=list)
    balls: list[dict[int, dict[str, Any]]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchTracks":
        return cls(
            players=data.get("players", []),
            referees=data.get("referees", []),
            balls=data.get("balls", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": self.players,
            "referees": self.referees,
            "balls": self.balls,
        }

    def __getitem__(self, item: str) -> list[dict[int, dict[str, Any]]]:
        if item == "players":
            return self.players
        elif item == "referees":
            return self.referees
        elif item == "balls":
            return self.balls
        raise KeyError(f"Invalid track entity key: {item}")

    def __setitem__(self, item: str, value: list[dict[int, dict[str, Any]]]) -> None:
        if item == "players":
            self.players = value
        elif item == "referees":
            self.referees = value
        elif item == "balls":
            self.balls = value
        else:
            raise KeyError(f"Invalid track entity key: {item}")

    def items(self):
        return [
            ("players", self.players),
            ("referees", self.referees),
            ("balls", self.balls),
        ]
