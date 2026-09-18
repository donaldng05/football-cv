"""
Typed configuration loader, models, and validation logic for football_cv.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    path: str = "models/best.pt"
    confidence: float = 0.10
    device: str = "auto"
    batch_size: int = 20

    def validate(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ConfigurationError(
                f"Model confidence must be in range [0.0, 1.0], got {self.confidence}"
            )
        if self.batch_size < 1:
            raise ConfigurationError(
                f"Model batch_size must be >= 1, got {self.batch_size}"
            )


@dataclass
class VideoConfig:
    input_path: str = "input_videos/08fd33_4.mp4"
    output_path: str = "output_videos/output_video.avi"
    frame_rate: float = 25.0
    start_frame: int = 0
    end_frame: int | None = None

    def validate(self) -> None:
        if self.frame_rate <= 0:
            raise ConfigurationError(
                f"Video frame_rate must be positive, got {self.frame_rate}"
            )
        if self.start_frame < 0:
            raise ConfigurationError(
                f"Video start_frame must be >= 0, got {self.start_frame}"
            )
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ConfigurationError(
                f"Video end_frame ({self.end_frame}) cannot be smaller than start_frame ({self.start_frame})"
            )


@dataclass
class TrackingConfig:
    use_cached_tracks: bool = True
    cache_path: str = "stubs/track_stubs.pkl"
    camera_movement_cache_path: str = "stubs/camera_movement_stubs.pkl"


@dataclass
class PossessionConfig:
    max_player_ball_distance: float = 70.0
    minimum_control_frames: int = 3
    maximum_missing_ball_frames: int = 10

    def validate(self) -> None:
        if self.max_player_ball_distance <= 0:
            raise ConfigurationError(
                f"max_player_ball_distance must be positive, got {self.max_player_ball_distance}"
            )
        if self.minimum_control_frames < 1:
            raise ConfigurationError(
                f"minimum_control_frames must be >= 1, got {self.minimum_control_frames}"
            )
        if self.maximum_missing_ball_frames < 0:
            raise ConfigurationError(
                f"maximum_missing_ball_frames must be >= 0, got {self.maximum_missing_ball_frames}"
            )


@dataclass
class MovementConfig:
    speed_window_frames: int = 5
    minimum_displacement: float = 0.0

    def validate(self) -> None:
        if self.speed_window_frames < 1:
            raise ConfigurationError(
                f"speed_window_frames must be >= 1, got {self.speed_window_frames}"
            )
        if self.minimum_displacement < 0:
            raise ConfigurationError(
                f"minimum_displacement must be >= 0.0, got {self.minimum_displacement}"
            )


@dataclass
class PerspectiveConfig:
    pixel_vertices: list[list[float]] = field(
        default_factory=lambda: [
            [110.0, 1035.0],
            [265.0, 275.0],
            [910.0, 260.0],
            [1640.0, 915.0],
        ]
    )
    court_width: float = 68.0
    court_length: float = 23.32

    def validate(self) -> None:
        if len(self.pixel_vertices) != 4:
            raise ConfigurationError(
                f"Perspective pixel_vertices must contain exactly 4 coordinate pairs, got {len(self.pixel_vertices)}"
            )
        for idx, vertex in enumerate(self.pixel_vertices):
            if len(vertex) != 2:
                raise ConfigurationError(
                    f"Pixel vertex at index {idx} must have [x, y], got {vertex}"
                )
        if self.court_width <= 0 or self.court_length <= 0:
            raise ConfigurationError("court_width and court_length must be positive")


@dataclass
class HeatmapConfig:
    enabled: bool = True
    grid_width: int = 60
    grid_height: int = 40

    def validate(self) -> None:
        if self.grid_width < 2 or self.grid_height < 2:
            raise ConfigurationError("Heatmap grid dimensions must be >= 2")


@dataclass
class PassNetworkConfig:
    enabled: bool = True
    maximum_transition_frames: int = 15

    def validate(self) -> None:
        if self.maximum_transition_frames < 1:
            raise ConfigurationError("maximum_transition_frames must be >= 1")


@dataclass
class AnalyticsConfig:
    enabled: bool = True
    export_events: bool = True
    export_dir: str = "outputs/analytics"
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    pass_network: PassNetworkConfig = field(default_factory=PassNetworkConfig)

    def validate(self) -> None:
        self.heatmap.validate()
        self.pass_network.validate()


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str | None = "outputs/logs/football-cv.log"

    def validate(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid logging level '{self.level}'. Must be one of {valid_levels}"
            )


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        self.model.validate()
        self.video.validate()
        self.possession.validate()
        self.movement.validate()
        self.perspective.validate()
        self.analytics.validate()
        self.logging.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """
    Load, merge, and validate football_cv configuration.

    Order of precedence:
      Overrides > Specified YAML > Default YAML (if exists) > Default dataclasses

    Args:
        config_path: Path to a YAML configuration file.
        overrides: Optional dictionary of parameter overrides.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigurationError: If the YAML is invalid or parameters fail validation.
    """
    raw_data: dict[str, Any] = {}

    target_path: Path | None = None
    if config_path is not None:
        target_path = Path(config_path)
        if not target_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
    elif Path("configs/default.yaml").is_file():
        target_path = Path("configs/default.yaml")

    if target_path is not None:
        try:
            with open(target_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    if not isinstance(loaded, dict):
                        raise ConfigurationError(
                            f"Configuration at {target_path} must be a dictionary"
                        )
                    raw_data = loaded
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse YAML file {target_path}: {exc}"
            ) from exc

    if overrides:
        _deep_update(raw_data, overrides)

    try:
        model_cfg = ModelConfig(**raw_data.get("model", {}))
        video_cfg = VideoConfig(**raw_data.get("video", {}))
        tracking_cfg = TrackingConfig(**raw_data.get("tracking", {}))
        possession_cfg = PossessionConfig(**raw_data.get("possession", {}))
        movement_cfg = MovementConfig(**raw_data.get("movement", {}))

        persp_data = raw_data.get("perspective", {})
        perspective_cfg = PerspectiveConfig(**persp_data)

        analytics_data = raw_data.get("analytics", {})
        heatmap_data = analytics_data.get("heatmap", {})
        pass_net_data = analytics_data.get("pass_network", {})

        heatmap_cfg = HeatmapConfig(**heatmap_data)
        pass_net_cfg = PassNetworkConfig(**pass_net_data)

        analytics_cfg = AnalyticsConfig(
            enabled=analytics_data.get("enabled", True),
            export_events=analytics_data.get("export_events", True),
            export_dir=analytics_data.get("export_dir", "outputs/analytics"),
            heatmap=heatmap_cfg,
            pass_network=pass_net_cfg,
        )

        logging_cfg = LoggingConfig(**raw_data.get("logging", {}))

        config = AppConfig(
            model=model_cfg,
            video=video_cfg,
            tracking=tracking_cfg,
            possession=possession_cfg,
            movement=movement_cfg,
            perspective=perspective_cfg,
            analytics=analytics_cfg,
            logging=logging_cfg,
        )
        config.validate()
        return config

    except TypeError as exc:
        raise ConfigurationError(f"Unexpected configuration parameter: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to construct configuration: {exc}") from exc
