"""
Tactical football pitch rendering utility using Matplotlib.

Supports standard FIFA pitch dimensions (105m x 68m) or custom metric layouts
with customizable visual themes (tactical dark, classic turf, clean light).
"""

from typing import Any

import matplotlib

matplotlib.use("Agg")  # Ensure headless compatibility
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

THEMES: dict[str, dict[str, str]] = {
    "tactical_dark": {
        "pitch_color": "#141b22",
        "line_color": "#445566",
        "text_color": "#e6edf3",
        "grid_color": "#1f2937",
    },
    "classic_turf": {
        "pitch_color": "#1f5628",
        "line_color": "#ffffff",
        "text_color": "#ffffff",
        "grid_color": "#17431f",
    },
    "light": {
        "pitch_color": "#f8f9fa",
        "line_color": "#343a40",
        "text_color": "#212529",
        "grid_color": "#e9ecef",
    },
}


def draw_pitch(
    ax: plt.Axes | None = None,
    length: float = 105.0,
    width: float = 68.0,
    theme: str = "tactical_dark",
    line_color: str | None = None,
    pitch_color: str | None = None,
    line_width: float = 1.5,
    show_padding: bool = True,
    padding: float = 4.0,
    draw_goals: bool = True,
) -> plt.Axes:
    """
    Draw a dimensionally accurate 2D tactical football pitch on a Matplotlib axes.

    Args:
        ax: Target matplotlib axes. If None, current axes will be used.
        length: Pitch length in meters (default: 105.0m).
        width: Pitch width in meters (default: 68.0m).
        theme: Visual palette theme ("tactical_dark", "classic_turf", "light").
        line_color: Custom line color override.
        pitch_color: Custom pitch background color override.
        line_width: Stroke width for pitch markings.
        show_padding: Whether to show run-off margin around the pitch.
        padding: Run-off margin size in meters.
        draw_goals: Whether to draw goal nets outside the touchlines.

    Returns:
        The matplotlib axes containing the pitch drawing.
    """
    if ax is None:
        ax = plt.gca()

    palette = THEMES.get(theme, THEMES["tactical_dark"])
    p_color = pitch_color or palette["pitch_color"]
    l_color = line_color or palette["line_color"]

    ax.set_facecolor(p_color)

    # Common patch arguments
    line_kwargs: dict[str, Any] = {
        "edgecolor": l_color,
        "linewidth": line_width,
        "zorder": 1,
    }

    # 1. Outer boundary rectangle
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            length,
            width,
            fill=False,
            **line_kwargs,
        )
    )

    # 2. Halfway line
    ax.plot(
        [length / 2, length / 2],
        [0.0, width],
        color=l_color,
        linewidth=line_width,
        zorder=1,
    )

    # 3. Center circle (radius 9.15m) & Center spot
    center_x = length / 2
    center_y = width / 2
    ax.add_patch(
        Circle(
            (center_x, center_y),
            9.15,
            fill=False,
            **line_kwargs,
        )
    )
    ax.add_patch(
        Circle(
            (center_x, center_y),
            0.4,
            facecolor=l_color,
            edgecolor=l_color,
            zorder=1,
        )
    )

    # 4. Penalty boxes (16.5m length, 40.32m width)
    box_half_width = 20.16
    box_y_start = center_y - box_half_width
    box_height = box_half_width * 2

    # Left penalty box
    ax.add_patch(
        Rectangle(
            (0.0, box_y_start),
            16.5,
            box_height,
            fill=False,
            **line_kwargs,
        )
    )
    # Right penalty box
    ax.add_patch(
        Rectangle(
            (length - 16.5, box_y_start),
            16.5,
            box_height,
            fill=False,
            **line_kwargs,
        )
    )

    # 5. Goal boxes (5.5m length, 18.32m width)
    goal_box_half_width = 9.16
    goal_box_y_start = center_y - goal_box_half_width
    goal_box_height = goal_box_half_width * 2

    # Left goal box
    ax.add_patch(
        Rectangle(
            (0.0, goal_box_y_start),
            5.5,
            goal_box_height,
            fill=False,
            **line_kwargs,
        )
    )
    # Right goal box
    ax.add_patch(
        Rectangle(
            (length - 5.5, goal_box_y_start),
            5.5,
            goal_box_height,
            fill=False,
            **line_kwargs,
        )
    )

    # 6. Penalty spots (11.0m from goal line)
    ax.add_patch(
        Circle(
            (11.0, center_y),
            0.4,
            facecolor=l_color,
            edgecolor=l_color,
            zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            (length - 11.0, center_y),
            0.4,
            facecolor=l_color,
            edgecolor=l_color,
            zorder=1,
        )
    )

    # 7. Penalty arcs (D outside 18-yard box, radius 9.15m from penalty spot)
    # cos(theta) = 5.5 / 9.15 ~= 0.601 -> theta ~= 53.0 degrees
    ax.add_patch(
        Arc(
            (11.0, center_y),
            width=18.3,
            height=18.3,
            angle=0.0,
            theta1=307.0,
            theta2=53.0,
            fill=False,
            **line_kwargs,
        )
    )
    ax.add_patch(
        Arc(
            (length - 11.0, center_y),
            width=18.3,
            height=18.3,
            angle=0.0,
            theta1=127.0,
            theta2=233.0,
            fill=False,
            **line_kwargs,
        )
    )

    # 8. Corner arcs (radius 1.0m)
    corner_radius = 1.0
    # Bottom-left (0, 0)
    ax.add_patch(
        Arc(
            (0.0, 0.0),
            width=corner_radius * 2,
            height=corner_radius * 2,
            angle=0.0,
            theta1=0.0,
            theta2=90.0,
            fill=False,
            **line_kwargs,
        )
    )
    # Top-left (0, width)
    ax.add_patch(
        Arc(
            (0.0, width),
            width=corner_radius * 2,
            height=corner_radius * 2,
            angle=0.0,
            theta1=270.0,
            theta2=360.0,
            fill=False,
            **line_kwargs,
        )
    )
    # Bottom-right (length, 0)
    ax.add_patch(
        Arc(
            (length, 0.0),
            width=corner_radius * 2,
            height=corner_radius * 2,
            angle=0.0,
            theta1=90.0,
            theta2=180.0,
            fill=False,
            **line_kwargs,
        )
    )
    # Top-right (length, width)
    ax.add_patch(
        Arc(
            (length, width),
            width=corner_radius * 2,
            height=corner_radius * 2,
            angle=0.0,
            theta1=180.0,
            theta2=270.0,
            fill=False,
            **line_kwargs,
        )
    )

    # 9. Goals (7.32m width, 2.0m depth)
    if draw_goals:
        goal_width = 7.32
        goal_depth = 2.0
        goal_y_start = center_y - goal_width / 2
        # Left goal
        ax.add_patch(
            Rectangle(
                (-goal_depth, goal_y_start),
                goal_depth,
                goal_width,
                fill=False,
                edgecolor=l_color,
                linewidth=line_width * 0.8,
                linestyle="--",
                alpha=0.6,
                zorder=1,
            )
        )
        # Right goal
        ax.add_patch(
            Rectangle(
                (length, goal_y_start),
                goal_depth,
                goal_width,
                fill=False,
                edgecolor=l_color,
                linewidth=line_width * 0.8,
                linestyle="--",
                alpha=0.6,
                zorder=1,
            )
        )

    # Formatting limits and aspect ratio
    pad = padding if show_padding else 0.0
    x_min = -pad - (2.5 if draw_goals and show_padding else 0.0)
    x_max = length + pad + (2.5 if draw_goals and show_padding else 0.0)
    y_min = -pad
    y_max = width + pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def create_pitch_figure(
    length: float = 105.0,
    width: float = 68.0,
    theme: str = "tactical_dark",
    figsize: tuple[float, float] = (12.0, 8.0),
    **draw_kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience helper to create a Matplotlib figure and pre-configured pitch axes.

    Args:
        length: Pitch length in meters.
        width: Pitch width in meters.
        theme: Visual theme name.
        figsize: Matplotlib figure dimensions (width, height) in inches.
        **draw_kwargs: Additional arguments forwarded to draw_pitch.

    Returns:
        Tuple of (Figure, Axes).
    """
    palette = THEMES.get(theme, THEMES["tactical_dark"])
    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["pitch_color"])
    draw_pitch(ax=ax, length=length, width=width, theme=theme, **draw_kwargs)
    return fig, ax
