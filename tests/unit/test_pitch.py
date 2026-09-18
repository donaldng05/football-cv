"""
Unit tests for tactical pitch drawing utility.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from football_cv.analytics.pitch import THEMES, create_pitch_figure, draw_pitch


class TestPitchDrawing:
    """Test suite for draw_pitch and pitch creation helpers."""

    def test_draw_pitch_defaults(self) -> None:
        fig, ax = plt.subplots()
        drawn_ax = draw_pitch(ax=ax)
        assert drawn_ax is ax

        # Verify patches added: outer rect, center circle, center spot, 2 penalty boxes,
        # 2 goal boxes, 2 penalty spots, 2 penalty arcs, 4 corner arcs, 2 goals
        assert len(ax.patches) >= 10

        # Check bounds with default 4m padding and goal margin
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] < 0.0
        assert xlim[1] > 105.0
        assert ylim[0] < 0.0
        assert ylim[1] > 68.0

        plt.close(fig)

    def test_draw_pitch_custom_dimensions(self) -> None:
        fig, ax = plt.subplots()
        draw_pitch(
            ax=ax,
            length=100.0,
            width=64.0,
            show_padding=False,
            draw_goals=False,
        )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (0.0, 100.0)
        assert ylim == (0.0, 64.0)

        plt.close(fig)

    def test_draw_pitch_themes(self) -> None:
        for theme_name, theme_data in THEMES.items():
            fig, ax = plt.subplots()
            draw_pitch(ax=ax, theme=theme_name)
            facecolor_hex = matplotlib.colors.to_hex(ax.get_facecolor())
            assert facecolor_hex.lower() == theme_data["pitch_color"].lower()
            plt.close(fig)

    def test_create_pitch_figure(self) -> None:
        fig, ax = create_pitch_figure(
            length=105.0,
            width=68.0,
            theme="classic_turf",
            figsize=(10.0, 7.0),
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_aspect() == 1.0  # Equal aspect ratio
        plt.close(fig)
