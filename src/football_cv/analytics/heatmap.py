"""
Possession Heatmap module for football_cv.

Generates explainable 2D spatial density visualizations of on-ball dominance
for teams and individual players, duration-weighted by frame duration (dt = 1 / fps).
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Ensure headless compatibility
import matplotlib.pyplot as plt
import numpy as np

from ..config import AppConfig, HeatmapConfig
from ..exceptions import AnalyticsError
from .pitch import THEMES, draw_pitch

logger = logging.getLogger(__name__)


@dataclass
class PossessionGrid:
    """Aggregated possession density grid on pitch coordinates."""

    raw_grid: np.ndarray  # Shape: (grid_height, grid_width)
    smoothed_grid: np.ndarray  # Shape: (grid_height, grid_width)
    total_possession_seconds: float
    total_frames: int
    xedges: np.ndarray
    yedges: np.ndarray
    pitch_length: float
    pitch_width: float
    grid_width: int
    grid_height: int
    fps: float


def apply_gaussian_smoothing(grid: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply 2D Gaussian smoothing to a grid.

    Uses scipy.ndimage.gaussian_filter when available, with a separable
    1D Gaussian convolution fallback.
    """
    if sigma <= 0.0:
        return grid.copy()

    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(grid, sigma=sigma, mode="nearest")
    except ImportError:
        radius = round(3 * sigma)
        if radius < 1:
            return grid.copy()
        x = np.arange(-radius, radius + 1)
        kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()

        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel_1d, mode="same"), axis=0, arr=grid
        )
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel_1d, mode="same"), axis=1, arr=smoothed
        )
        return smoothed


def filter_possession_records(
    records: list[dict[str, Any]],
    team_id: int | None = None,
    player_id: int | None = None,
    min_confidence: float = 0.0,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[dict[str, Any]]:
    """
    Filter tracking records strictly for qualified on-ball possession.

    Criteria:
      - has_ball is True
      - x_pitch and y_pitch are valid numeric values
      - matches team_id (if specified)
      - matches player_id (if specified)
      - detection_confidence >= min_confidence
      - frame_index within [start_frame, end_frame] (if specified)

    Args:
        records: List of tracking record dictionaries.
        team_id: Optional team ID filter (e.g. 1 or 2).
        player_id: Optional player track ID filter.
        min_confidence: Minimum detection or possession confidence.
        start_frame: Optional lower frame index bound.
        end_frame: Optional upper frame index bound.

    Returns:
        Filtered list of possession record dictionaries.
    """
    qualified: list[dict[str, Any]] = []

    for r in records:
        # Possession check
        has_ball = r.get("has_ball")
        if isinstance(has_ball, str):
            has_ball = has_ball.strip().lower() in {"true", "1", "yes"}
        elif not bool(has_ball):
            continue

        # Coordinate checks
        x_val = r.get("x_pitch")
        y_val = r.get("y_pitch")
        if x_val is None or y_val is None:
            continue
        try:
            x_f = float(x_val)
            y_f = float(y_val)
        except (ValueError, TypeError):
            continue

        if np.isnan(x_f) or np.isnan(y_f) or np.isinf(x_f) or np.isinf(y_f):
            continue

        # Confidence filter
        conf = r.get("detection_confidence", r.get("confidence", 1.0))
        try:
            conf_f = float(conf)
        except (ValueError, TypeError):
            conf_f = 1.0

        if conf_f < min_confidence:
            continue

        # Frame range filter
        frame_idx = r.get("frame_index")
        if frame_idx is not None:
            try:
                f_int = int(frame_idx)
                if start_frame is not None and f_int < start_frame:
                    continue
                if end_frame is not None and f_int > end_frame:
                    continue
            except (ValueError, TypeError):
                pass

        # Team filter
        if team_id is not None:
            r_team = r.get("team_id", r.get("team"))
            if r_team is None:
                continue
            try:
                if int(r_team) != int(team_id):
                    continue
            except (ValueError, TypeError):
                continue

        # Player filter
        if player_id is not None:
            r_player = r.get("player_id")
            if r_player is None:
                continue
            try:
                if int(r_player) != int(player_id):
                    continue
            except (ValueError, TypeError):
                continue

        qualified.append(r)

    return qualified


def aggregate_possession_grid(
    records: list[dict[str, Any]],
    grid_width: int = 60,
    grid_height: int = 40,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float = 25.0,
    sigma: float = 1.5,
) -> PossessionGrid:
    """
    Bin possession observations into a 2D spatial grid weighted by frame duration (dt = 1 / fps).

    Args:
        records: Qualified possession records with x_pitch and y_pitch coordinates.
        grid_width: Number of bins along the pitch length (X axis).
        grid_height: Number of bins along the pitch width (Y axis).
        pitch_length: Pitch length in meters (default: 105.0m).
        pitch_width: Pitch width in meters (default: 68.0m).
        fps: Frames per second used for duration weighting.
        sigma: Standard deviation for 2D Gaussian smoothing.

    Returns:
        PossessionGrid containing raw duration grid, smoothed grid, and metadata.
    """
    if grid_width < 2 or grid_height < 2:
        raise AnalyticsError(
            f"Heatmap grid dimensions must be >= 2, got ({grid_width}, {grid_height})"
        )
    if pitch_length <= 0 or pitch_width <= 0:
        raise AnalyticsError(
            f"Pitch dimensions must be > 0, got ({pitch_length}, {pitch_width})"
        )

    dt = 1.0 / fps if fps > 0 else 0.04

    xedges = np.linspace(0.0, pitch_length, grid_width + 1)
    yedges = np.linspace(0.0, pitch_width, grid_height + 1)

    if not records:
        raw_grid = np.zeros((grid_height, grid_width), dtype=np.float64)
        return PossessionGrid(
            raw_grid=raw_grid,
            smoothed_grid=raw_grid.copy(),
            total_possession_seconds=0.0,
            total_frames=0,
            xedges=xedges,
            yedges=yedges,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            grid_width=grid_width,
            grid_height=grid_height,
            fps=fps,
        )

    x_coords = np.array([float(r["x_pitch"]) for r in records], dtype=np.float64)
    y_coords = np.array([float(r["y_pitch"]) for r in records], dtype=np.float64)

    # Duration weights: each observation corresponds to dt seconds
    weights = np.full(len(records), dt, dtype=np.float64)

    # 2D histogram: X on axis 0, Y on axis 1
    # We clip coordinates within pitch boundary [0, length] and [0, width]
    x_coords = np.clip(x_coords, 0.0, pitch_length - 1e-5)
    y_coords = np.clip(y_coords, 0.0, pitch_width - 1e-5)

    hist, _, _ = np.histogram2d(
        x_coords,
        y_coords,
        bins=[xedges, yedges],
        weights=weights,
    )

    # Transpose so rows correspond to Y (height) and columns correspond to X (width)
    raw_grid = hist.T

    # Apply 2D Gaussian smoothing
    smoothed_grid = apply_gaussian_smoothing(raw_grid, sigma=sigma)

    total_seconds = float(np.sum(raw_grid))
    total_frames = len(records)

    return PossessionGrid(
        raw_grid=raw_grid,
        smoothed_grid=smoothed_grid,
        total_possession_seconds=round(total_seconds, 3),
        total_frames=total_frames,
        xedges=xedges,
        yedges=yedges,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        grid_width=grid_width,
        grid_height=grid_height,
        fps=fps,
    )


def render_possession_heatmap(
    grid: PossessionGrid,
    output_path: str | Path | None = None,
    title: str = "Possession Heatmap",
    subtitle: str | None = None,
    colormap: str = "inferno",
    theme: str = "tactical_dark",
    alpha: float = 0.80,
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a possession heatmap overlaid on a 2D tactical pitch.

    Args:
        grid: PossessionGrid data structure.
        output_path: Optional path to write PNG file.
        title: Main figure title.
        subtitle: Secondary subtitle line (e.g. duration, team name).
        colormap: Matplotlib colormap name (e.g. 'inferno', 'YlOrRd', 'plasma').
        theme: Tactical pitch theme ('tactical_dark', 'classic_turf', 'light').
        alpha: Maximum transparency opacity for high-density areas.
        figsize: Output figure dimensions.
        dpi: Output resolution.

    Returns:
        Tuple of (Figure, Axes).
    """
    palette = THEMES.get(theme, THEMES["tactical_dark"])

    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["pitch_color"])

    # Draw tactical pitch
    draw_pitch(
        ax=ax,
        length=grid.pitch_length,
        width=grid.pitch_width,
        theme=theme,
    )

    # Prepare RGBA heatmap overlay with smooth alpha gradient
    max_val = float(grid.smoothed_grid.max())
    if max_val > 1e-6:
        norm = plt.Normalize(vmin=0.0, vmax=max_val)
        cmap = plt.get_cmap(colormap)
        rgba = cmap(norm(grid.smoothed_grid))

        # Dynamic alpha mask: zero intensity is fully transparent
        intensity = norm(grid.smoothed_grid)
        rgba[..., 3] = np.clip(intensity * 1.4, 0.0, 1.0) * alpha
        rgba[grid.smoothed_grid < 1e-5, 3] = 0.0

        # Overlay heatmap on pitch
        ax.imshow(
            rgba,
            extent=[0.0, grid.pitch_length, 0.0, grid.pitch_width],
            origin="lower",
            interpolation="bicubic",
            zorder=2,
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.040,
            pad=0.03,
            shrink=0.55,
        )
        cbar.set_label(
            "Possession Duration (seconds / bin)",
            color=palette["text_color"],
            fontsize=10,
        )
        cbar.ax.tick_params(colors=palette["text_color"], labelsize=9)
        cbar.outline.set_edgecolor(palette["line_color"])  # type: ignore[union-attr]

    # Title & Subtitle formatting
    ax.text(
        0.5,
        1.05,
        title,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=palette["text_color"],
        ha="center",
        va="bottom",
    )

    sub_text = subtitle
    if sub_text is None:
        sub_text = (
            f"Total Possession: {grid.total_possession_seconds:.1f}s | "
            f"Frames: {grid.total_frames} | "
            f"Grid: {grid.grid_width}x{grid.grid_height}"
        )

    ax.text(
        0.5,
        1.01,
        sub_text,
        transform=ax.transAxes,
        fontsize=10,
        color=palette["line_color"],
        ha="center",
        va="bottom",
    )

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            target,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=palette["pitch_color"],
        )
        logger.info(f"Saved possession heatmap to {target}")

    return fig, ax


def export_heatmap_csv(
    grid: PossessionGrid,
    output_path: str | Path,
    team_id: int | None = None,
    player_id: int | None = None,
    include_zero_cells: bool = False,
) -> Path:
    """
    Export aggregated grid density data to a structured CSV file.

    Args:
        grid: PossessionGrid to export.
        output_path: Target CSV file path.
        team_id: Optional team identifier tag.
        player_id: Optional player identifier tag.
        include_zero_cells: Whether to write empty zero-duration cells.

    Returns:
        Path to the written CSV file.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "team_id",
        "player_id",
        "grid_x_bin",
        "grid_y_bin",
        "x_pitch_center",
        "y_pitch_center",
        "possession_seconds",
        "smoothed_density",
        "density_share_percent",
    ]

    total_sec = (
        grid.total_possession_seconds if grid.total_possession_seconds > 0 else 1.0
    )

    x_centers = (grid.xedges[:-1] + grid.xedges[1:]) / 2.0
    y_centers = (grid.yedges[:-1] + grid.yedges[1:]) / 2.0

    rows: list[dict[str, Any]] = []

    for y_bin in range(grid.grid_height):
        for x_bin in range(grid.grid_width):
            raw_sec = float(grid.raw_grid[y_bin, x_bin])
            smooth_val = float(grid.smoothed_grid[y_bin, x_bin])

            if not include_zero_cells and raw_sec <= 0.0 and smooth_val <= 1e-4:
                continue

            share_pct = round((raw_sec / total_sec) * 100.0, 3)

            rows.append(
                {
                    "team_id": team_id,
                    "player_id": player_id,
                    "grid_x_bin": x_bin,
                    "grid_y_bin": y_bin,
                    "x_pitch_center": round(float(x_centers[x_bin]), 2),
                    "y_pitch_center": round(float(y_centers[y_bin]), 2),
                    "possession_seconds": round(raw_sec, 3),
                    "smoothed_density": round(smooth_val, 4),
                    "density_share_percent": share_pct,
                }
            )

    with open(target, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} heatmap grid cells to {target}")
    return target


class HeatmapGenerator:
    """
    High-level service for generating possession heatmaps and tabular exports.

    Ingests match tracking records from JSON, CSV, or in-memory lists,
    filters on-ball dominance, and renders team/player artifacts.
    """

    def __init__(
        self,
        config: AppConfig | HeatmapConfig | None = None,
        output_dir: str | Path = "outputs/report",
    ):
        if isinstance(config, AppConfig):
            self.heatmap_cfg = config.analytics.heatmap
            self.fps = config.video.frame_rate
        elif isinstance(config, HeatmapConfig):
            self.heatmap_cfg = config
            self.fps = 25.0
        else:
            self.heatmap_cfg = HeatmapConfig()
            self.fps = 25.0

        self.output_dir = Path(output_dir)
        self.heatmaps_dir = self.output_dir / "heatmaps"
        self.data_dir = self.output_dir / "data"

    def load_records_from_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Load player tracking records from a JSON or CSV file."""
        p = Path(file_path)
        if not p.is_file():
            raise AnalyticsError(f"Tracking file not found: {file_path}")

        if p.suffix.lower() == ".json":
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                raise AnalyticsError(
                    f"Expected list in tracking JSON, got {type(data)}"
                )
            except json.JSONDecodeError as exc:
                raise AnalyticsError(f"Invalid JSON file {p}: {exc}") from exc

        elif p.suffix.lower() == ".csv":
            records: list[dict[str, Any]] = []
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            return records

        raise AnalyticsError(
            f"Unsupported file format '{p.suffix}'. Expected .json or .csv"
        )

    def generate_from_records(
        self,
        records: list[dict[str, Any]],
        fps: float | None = None,
        team_filter: int | None = None,
        player_filter: int | None = None,
    ) -> dict[str, Path]:
        """
        Generate team heatmaps, top player heatmaps, and tabular data from records.

        Args:
            records: Tracking records.
            fps: Frame rate for duration weighting (falls back to config frame rate).
            team_filter: If specified, only generate for this team.
            player_filter: If specified, only generate for this player.

        Returns:
            Dictionary mapping artifact names to their generated file paths.
        """
        effective_fps = fps or self.fps
        artifacts: dict[str, Path] = {}

        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.heatmap_cfg

        # 1. Target player specific heatmap
        if player_filter is not None:
            p_records = filter_possession_records(
                records,
                player_id=player_filter,
                min_confidence=cfg.min_confidence,
            )
            p_grid = aggregate_possession_grid(
                p_records,
                grid_width=cfg.grid_width,
                grid_height=cfg.grid_height,
                pitch_length=cfg.pitch_length,
                pitch_width=cfg.pitch_width,
                fps=effective_fps,
                sigma=cfg.sigma,
            )
            out_img = self.heatmaps_dir / f"player_{player_filter}_possession.png"
            render_possession_heatmap(
                p_grid,
                output_path=out_img,
                title=f"Player {player_filter} - Possession Heatmap",
                subtitle=f"Possession: {p_grid.total_possession_seconds:.1f}s ({p_grid.total_frames} frames)",
                colormap=cfg.colormap,
                theme=cfg.theme,
            )
            plt.close("all")
            artifacts[f"player_{player_filter}"] = out_img
            return artifacts

        # 2. Team heatmaps
        teams_to_process = [team_filter] if team_filter is not None else [1, 2]

        for tid in teams_to_process:
            team_records = filter_possession_records(
                records,
                team_id=tid,
                min_confidence=cfg.min_confidence,
            )
            grid = aggregate_possession_grid(
                team_records,
                grid_width=cfg.grid_width,
                grid_height=cfg.grid_height,
                pitch_length=cfg.pitch_length,
                pitch_width=cfg.pitch_width,
                fps=effective_fps,
                sigma=cfg.sigma,
            )

            out_img = self.heatmaps_dir / f"team_{tid}_possession.png"
            render_possession_heatmap(
                grid,
                output_path=out_img,
                title=f"Team {tid} - Possession Dominance Heatmap",
                subtitle=f"Control Time: {grid.total_possession_seconds:.1f}s | On-Ball Frames: {grid.total_frames}",
                colormap=cfg.colormap,
                theme=cfg.theme,
            )
            plt.close("all")
            artifacts[f"team_{tid}"] = out_img

            # Accumulate CSV data
            if grid.total_possession_seconds > 0:
                team_csv_path = self.data_dir / f"possession_heatmap_team_{tid}.csv"
                export_heatmap_csv(grid, team_csv_path, team_id=tid)

        # 3. Overall possession heatmap & combined CSV
        overall_records = filter_possession_records(
            records,
            min_confidence=cfg.min_confidence,
        )
        combined_grid = aggregate_possession_grid(
            overall_records,
            grid_width=cfg.grid_width,
            grid_height=cfg.grid_height,
            pitch_length=cfg.pitch_length,
            pitch_width=cfg.pitch_width,
            fps=effective_fps,
            sigma=cfg.sigma,
        )
        combined_csv = self.data_dir / "possession_heatmap.csv"
        export_heatmap_csv(combined_grid, combined_csv)
        artifacts["heatmap_csv"] = combined_csv

        # 4. Prominent individual players
        # Group possession frames by player_id
        player_counts: dict[int, int] = {}
        for r in overall_records:
            pid = r.get("player_id")
            if pid is not None:
                try:
                    pid_int = int(pid)
                    player_counts[pid_int] = player_counts.get(pid_int, 0) + 1
                except (ValueError, TypeError):
                    pass

        # Top 3 players with at least 5 frames of possession
        sorted_players = sorted(
            [(p, count) for p, count in player_counts.items() if count >= 5],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        for pid, _ in sorted_players:
            p_records = filter_possession_records(
                records,
                player_id=pid,
                min_confidence=cfg.min_confidence,
            )
            p_grid = aggregate_possession_grid(
                p_records,
                grid_width=cfg.grid_width,
                grid_height=cfg.grid_height,
                pitch_length=cfg.pitch_length,
                pitch_width=cfg.pitch_width,
                fps=effective_fps,
                sigma=cfg.sigma,
            )
            out_img = self.heatmaps_dir / f"player_{pid}_possession.png"
            render_possession_heatmap(
                p_grid,
                output_path=out_img,
                title=f"Player {pid} - Possession Heatmap",
                subtitle=f"Possession: {p_grid.total_possession_seconds:.1f}s ({p_grid.total_frames} frames)",
                colormap=cfg.colormap,
                theme=cfg.theme,
            )
            plt.close("all")
            artifacts[f"player_{pid}"] = out_img

        return artifacts

    def generate_from_file(
        self,
        file_path: str | Path,
        fps: float | None = None,
        team_filter: int | None = None,
        player_filter: int | None = None,
    ) -> dict[str, Path]:
        """Convenience method to generate heatmaps directly from a data file."""
        records = self.load_records_from_file(file_path)
        return self.generate_from_records(
            records=records,
            fps=fps,
            team_filter=team_filter,
            player_filter=player_filter,
        )
