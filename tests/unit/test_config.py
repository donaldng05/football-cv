"""
Unit tests for football_cv configuration loading, schema validation, and overrides.
"""

import pytest

from football_cv.config import (
    AppConfig,
    ModelConfig,
    PerspectiveConfig,
    PossessionConfig,
    VideoConfig,
    load_config,
)
from football_cv.exceptions import ConfigurationError


class TestConfigLoading:
    def test_load_default_yaml_config(self):
        config = load_config("configs/default.yaml")
        assert isinstance(config, AppConfig)
        assert config.model.confidence == 0.10
        assert config.video.frame_rate == 25.0
        assert config.possession.max_player_ball_distance == 70.0
        assert len(config.perspective.pixel_vertices) == 4
        assert config.analytics.enabled is True
        assert config.vision.backend == "python"

    def test_load_fast_yaml_config(self):
        config = load_config("configs/fast.yaml")
        assert config.model.confidence == 0.20
        assert config.video.end_frame == 50
        assert config.model.device == "cpu"
        assert config.logging.level == "DEBUG"
        assert config.vision.backend == "python"

    def test_load_high_accuracy_yaml_config(self):
        config = load_config("configs/high_accuracy.yaml")
        assert config.model.confidence == 0.15
        assert config.model.device == "cuda"
        assert config.model.batch_size == 32
        assert config.vision.backend == "python"

    def test_missing_config_file_raises_error(self):
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("configs/non_existent_file.yaml")

    def test_malformed_yaml_raises_error(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("model: [unclosed list", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Failed to parse YAML file"):
            load_config(bad_yaml)

    def test_non_dict_yaml_raises_error(self, tmp_path):
        bad_yaml = tmp_path / "list.yaml"
        bad_yaml.write_text("- item1\n- item2", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must be a dictionary"):
            load_config(bad_yaml)


class TestConfigOverrides:
    def test_cli_overrides_precedence(self):
        overrides = {
            "model": {"confidence": 0.45, "device": "cpu"},
            "video": {"start_frame": 10, "end_frame": 20},
        }
        config = load_config("configs/default.yaml", overrides=overrides)
        assert config.model.confidence == 0.45
        assert config.model.device == "cpu"
        assert config.video.start_frame == 10
        assert config.video.end_frame == 20
        # Untouched values should retain YAML defaults
        assert config.model.batch_size == 20
        assert config.possession.max_player_ball_distance == 70.0

    def test_unexpected_override_key_raises_error(self):
        overrides = {"model": {"unknown_param_xyz": 123}}
        with pytest.raises(
            ConfigurationError, match="Unexpected configuration parameter"
        ):
            load_config("configs/default.yaml", overrides=overrides)


class TestConfigValidationRules:
    def test_invalid_model_confidence_raises(self):
        with pytest.raises(
            ConfigurationError, match="Model confidence must be in range"
        ):
            ModelConfig(confidence=1.5).validate()

        with pytest.raises(
            ConfigurationError, match="Model confidence must be in range"
        ):
            ModelConfig(confidence=-0.1).validate()

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ConfigurationError, match="batch_size must be >= 1"):
            ModelConfig(batch_size=0).validate()

    def test_invalid_frame_bounds_raises(self):
        with pytest.raises(
            ConfigurationError, match="cannot be smaller than start_frame"
        ):
            VideoConfig(start_frame=100, end_frame=50).validate()

        with pytest.raises(ConfigurationError, match="start_frame must be >= 0"):
            VideoConfig(start_frame=-5).validate()

    def test_invalid_possession_distance_raises(self):
        with pytest.raises(
            ConfigurationError, match="max_player_ball_distance must be positive"
        ):
            PossessionConfig(max_player_ball_distance=-10).validate()

    def test_invalid_perspective_vertices_raises(self):
        # 3 points instead of 4
        with pytest.raises(
            ConfigurationError, match="must contain exactly 4 coordinate pairs"
        ):
            PerspectiveConfig(pixel_vertices=[[0, 0], [10, 0], [10, 10]]).validate()

        # 3 coordinates in a point
        with pytest.raises(ConfigurationError, match="must have \\[x, y\\]"):
            PerspectiveConfig(
                pixel_vertices=[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
            ).validate()
