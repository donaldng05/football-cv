"""
Pass-Network Inference module for football_cv.

Infers team passing networks from tracked possession transitions and candidate events,
representing players as spatial centroid nodes and candidate passes as confidence-weighted
directed graph edges on a 2D tactical pitch.
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
from matplotlib.patches import Circle, FancyArrowPatch

from ..config import AppConfig, PassNetworkConfig
from ..exceptions import AnalyticsError
from .event_builder import CandidateEvent, EventType
from .pitch import THEMES, draw_pitch

logger = logging.getLogger(__name__)


@dataclass
class PassNode:
    """Represents a player track node in a tactical pass network."""

    player_id: int
    team_id: int
    x_centroid: float
    y_centroid: float
    possession_seconds: float
    passes_sent: int
    passes_received: int
    involvement_count: int


@dataclass
class PassEdge:
    """Represents a directed candidate pass connection between two teammate tracks."""

    from_player_id: int
    to_player_id: int
    team_id: int
    pass_count: int
    avg_transition_seconds: float
    avg_confidence: float


@dataclass
class PassNetwork:
    """Container for a team's complete tactical pass network graph."""

    team_id: int
    nodes: dict[int, PassNode]
    edges: list[PassEdge]
    total_passes: int
    pitch_length: float = 105.0
    pitch_width: float = 68.0


def build_pass_network(
    events: list[CandidateEvent | dict[str, Any]],
    intervals: list[Any] | None = None,
    player_records: list[dict[str, Any]] | None = None,
    team_id: int = 1,
    min_passes: int = 1,
    min_confidence: float = 0.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> PassNetwork:
    """
    Build a directed passing network graph for a given team from candidate pass events.

    Args:
        events: List of candidate events (CandidateEvent instances or dicts).
        intervals: Optional list of possession intervals for duration calculation.
        player_records: Optional list of frame-by-frame player tracking records.
        team_id: Target team ID (e.g. 1 or 2).
        min_passes: Minimum candidate pass volume required to draw an edge.
        min_confidence: Minimum candidate pass confidence threshold.
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Returns:
        PassNetwork instance containing computed nodes and directed edges.
    """
    if min_passes < 1:
        raise AnalyticsError(f"min_passes must be >= 1, got {min_passes}")
    if pitch_length <= 0 or pitch_width <= 0:
        raise AnalyticsError("Pitch dimensions must be > 0")

    # 1. Filter events for candidate passes belonging to the target team
    qualified_passes: list[dict[str, Any]] = []
    for ev in events:
        d = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)

        ev_type = d.get("event_type")
        if ev_type != EventType.CANDIDATE_PASS.value and ev_type != "candidate_pass":
            continue

        f_team = d.get("from_team_id")
        t_team = d.get("to_team_id")
        if f_team is None or t_team is None:
            continue
        if int(f_team) != team_id or int(t_team) != team_id:
            continue

        f_pid = d.get("from_player_id")
        t_pid = d.get("to_player_id")
        if f_pid is None or t_pid is None:
            continue
        if int(f_pid) == int(t_pid):  # Ignore self-transitions
            continue

        conf = float(d.get("confidence", 1.0))
        if conf < min_confidence:
            continue

        qualified_passes.append(d)

    # 2. Accumulate spatial coordinates and durations for nodes
    player_x_coords: dict[int, list[float]] = {}
    player_y_coords: dict[int, list[float]] = {}
    player_durations: dict[int, float] = {}
    passes_sent: dict[int, int] = {}
    passes_received: dict[int, int] = {}

    # Extract coordinates from candidate pass start/end points
    for p in qualified_passes:
        from_id = int(p["from_player_id"])
        to_id = int(p["to_player_id"])

        passes_sent[from_id] = passes_sent.get(from_id, 0) + 1
        passes_received[to_id] = passes_received.get(to_id, 0) + 1

        sx = p.get("start_x_pitch")
        sy = p.get("start_y_pitch")
        if sx is not None and sy is not None:
            player_x_coords.setdefault(from_id, []).append(float(sx))
            player_y_coords.setdefault(from_id, []).append(float(sy))

        ex = p.get("end_x_pitch")
        ey = p.get("end_y_pitch")
        if ex is not None and ey is not None:
            player_x_coords.setdefault(to_id, []).append(float(ex))
            player_y_coords.setdefault(to_id, []).append(float(ey))

    # Incorporate possession intervals if available for accurate duration
    if intervals:
        for itv in intervals:
            itv_d = itv.to_dict() if hasattr(itv, "to_dict") else dict(itv)
            if int(itv_d.get("team_id", -1)) == team_id:
                pid = int(itv_d.get("player_id", -1))
                if pid > 0:
                    dur = float(itv_d.get("duration_seconds", 0.0))
                    player_durations[pid] = player_durations.get(pid, 0.0) + dur

    # Incorporate frame player records for more refined centroid locations
    if player_records:
        for r in player_records:
            r_team = r.get("team_id", r.get("team"))
            if r_team is not None and int(r_team) == team_id:
                pid = r.get("player_id")
                has_ball = r.get("has_ball")
                if pid is not None and (
                    bool(has_ball)
                    or int(pid) in passes_sent
                    or int(pid) in passes_received
                ):
                    xp = r.get("x_pitch")
                    yp = r.get("y_pitch")
                    if xp is not None and yp is not None:
                        player_x_coords.setdefault(int(pid), []).append(float(xp))
                        player_y_coords.setdefault(int(pid), []).append(float(yp))

    # All active players in this network
    all_pids = set(passes_sent.keys()) | set(passes_received.keys())
    nodes: dict[int, PassNode] = {}

    for pid in all_pids:
        xs = player_x_coords.get(pid, [])
        ys = player_y_coords.get(pid, [])

        x_c = float(np.mean(xs)) if xs else pitch_length / 2.0
        y_c = float(np.mean(ys)) if ys else pitch_width / 2.0

        # Bound centroid within pitch margins
        x_c = float(np.clip(x_c, 5.0, pitch_length - 5.0))
        y_c = float(np.clip(y_c, 5.0, pitch_width - 5.0))

        sent = passes_sent.get(pid, 0)
        recv = passes_received.get(pid, 0)
        dur = round(player_durations.get(pid, (sent + recv) * 0.8), 2)

        nodes[pid] = PassNode(
            player_id=pid,
            team_id=team_id,
            x_centroid=round(x_c, 2),
            y_centroid=round(y_c, 2),
            possession_seconds=dur,
            passes_sent=sent,
            passes_received=recv,
            involvement_count=sent + recv,
        )

    # 3. Aggregate directed edges between (from_player, to_player)
    edge_buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for p in qualified_passes:
        u = int(p["from_player_id"])
        v = int(p["to_player_id"])
        edge_buckets.setdefault((u, v), []).append(p)

    edges: list[PassEdge] = []
    total_passes = len(qualified_passes)

    for (u, v), p_list in edge_buckets.items():
        if len(p_list) < min_passes:
            continue

        conf_sum = sum(float(p.get("confidence", 1.0)) for p in p_list)
        time_sum = 0.0
        for p in p_list:
            s_time = float(p.get("start_time", 0.0))
            e_time = float(p.get("end_time", s_time))
            time_sum += max(0.0, e_time - s_time)

        edges.append(
            PassEdge(
                from_player_id=u,
                to_player_id=v,
                team_id=team_id,
                pass_count=len(p_list),
                avg_transition_seconds=round(time_sum / len(p_list), 2),
                avg_confidence=round(conf_sum / len(p_list), 3),
            )
        )

    # Sort edges by pass count descending
    edges.sort(key=lambda e: e.pass_count, reverse=True)

    return PassNetwork(
        team_id=team_id,
        nodes=nodes,
        edges=edges,
        total_passes=total_passes,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )


def render_pass_network(
    network: PassNetwork,
    output_path: str | Path | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "tactical_dark",
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 150,
    node_scale: float = 1.0,
    edge_scale: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render directed passing network topology on a 2D tactical pitch.

    Args:
        network: PassNetwork instance.
        output_path: Optional file path to save PNG image.
        title: Main figure title.
        subtitle: Optional secondary subtitle.
        theme: Tactical pitch theme ('tactical_dark', 'classic_turf', 'light').
        figsize: Figure dimensions in inches.
        dpi: Resolution of exported figure.
        node_scale: Multiplier for node circle radii.
        edge_scale: Multiplier for directed arrow line thickness.

    Returns:
        Tuple of (Figure, Axes).
    """
    palette = THEMES.get(theme, THEMES["tactical_dark"])

    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["pitch_color"])

    # Draw tactical pitch
    draw_pitch(
        ax=ax,
        length=network.pitch_length,
        width=network.pitch_width,
        theme=theme,
    )

    # Team palette accents
    node_color = "#38bdf8" if network.team_id == 1 else "#f97316"
    edge_color = "#7dd3fc" if network.team_id == 1 else "#fdba74"
    text_color = "#0f172a" if theme == "light" else "#ffffff"

    # 1. Precompute node radii for arrow endpoint clipping
    max_involvement = max(
        [n.involvement_count for n in network.nodes.values()], default=1
    )
    node_radii: dict[int, float] = {}
    for pid, node in network.nodes.items():
        inv_ratio = node.involvement_count / max(1, max_involvement)
        node_radii[pid] = (1.8 + np.sqrt(inv_ratio) * 1.6) * node_scale

    # 2. Draw directed edges with visible directional arrowheads
    max_passes = max([e.pass_count for e in network.edges], default=1)

    for edge in network.edges:
        if (
            edge.from_player_id not in network.nodes
            or edge.to_player_id not in network.nodes
        ):
            continue

        u_node = network.nodes[edge.from_player_id]
        v_node = network.nodes[edge.to_player_id]

        x1, y1 = u_node.x_centroid, u_node.y_centroid
        x2, y2 = v_node.x_centroid, v_node.y_centroid

        dx = x2 - x1
        dy = y2 - y1
        dist = float(np.hypot(dx, dy))
        if dist < 1e-3:
            continue

        ux = dx / dist
        uy = dy / dist

        r1 = node_radii.get(edge.from_player_id, 2.0 * node_scale)
        r2 = node_radii.get(edge.to_player_id, 2.0 * node_scale)

        # Offset start and end so arrow starts outside source and ends cleanly outside target
        sx = x1 + ux * (r1 + 0.3)
        sy = y1 + uy * (r1 + 0.3)
        ex = x2 - ux * (r2 + 0.4)
        ey = y2 - uy * (r2 + 0.4)

        # Normalized stroke width (1.5 to 6.0 px)
        weight_norm = edge.pass_count / max(1, max_passes)
        lw = (1.5 + weight_norm * 4.5) * edge_scale

        # Opacity mapped to transition confidence (0.45 to 0.95)
        alpha = float(np.clip(edge.avg_confidence * 0.95, 0.45, 0.95))

        # Curved directed arrow with crisp visible arrowhead
        arrow = FancyArrowPatch(
            (sx, sy),
            (ex, ey),
            connectionstyle="arc3,rad=0.15",
            arrowstyle="-|>",
            mutation_scale=13 + weight_norm * 4,
            color=edge_color,
            linewidth=lw,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(arrow)

    # 3. Draw nodes and player track labels
    for pid, node in network.nodes.items():
        radius = node_radii[pid]

        # Outer glowing ring
        ax.add_patch(
            Circle(
                (node.x_centroid, node.y_centroid),
                radius + 0.3,
                facecolor="none",
                edgecolor=node_color,
                linewidth=1.2,
                alpha=0.6,
                zorder=3,
            )
        )

        # Main node circle
        ax.add_patch(
            Circle(
                (node.x_centroid, node.y_centroid),
                radius,
                facecolor=node_color,
                edgecolor="#ffffff",
                linewidth=1.5,
                zorder=4,
            )
        )

        # Track ID label
        ax.text(
            node.x_centroid,
            node.y_centroid,
            f"#{node.player_id}",
            color=text_color,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )

    # 3. Header & Metadata
    main_title = title or f"Team {network.team_id} - Passing Network"
    sub_title = (
        subtitle
        or f"Candidate Passes: {network.total_passes} | Connections: {len(network.edges)} | Active Tracks: {len(network.nodes)}"
    )

    ax.text(
        0.5,
        1.05,
        main_title,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=palette["text_color"],
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5,
        1.01,
        sub_title,
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
        logger.info(f"Saved pass network visualization to {target}")

    return fig, ax


def export_pass_network_csv(
    network: PassNetwork,
    nodes_csv_path: str | Path,
    edges_csv_path: str | Path,
) -> tuple[Path, Path]:
    """
    Export pass network graph nodes and directed edges to structured CSV files.

    Returns:
        Tuple of (nodes_csv_path, edges_csv_path).
    """
    n_path = Path(nodes_csv_path)
    e_path = Path(edges_csv_path)
    n_path.parent.mkdir(parents=True, exist_ok=True)
    e_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export Nodes CSV
    node_fields = [
        "team_id",
        "player_id",
        "x_pitch_centroid",
        "y_pitch_centroid",
        "possession_seconds",
        "passes_sent",
        "passes_received",
        "involvement_count",
    ]

    node_rows = [
        {
            "team_id": n.team_id,
            "player_id": n.player_id,
            "x_pitch_centroid": n.x_centroid,
            "y_pitch_centroid": n.y_centroid,
            "possession_seconds": n.possession_seconds,
            "passes_sent": n.passes_sent,
            "passes_received": n.passes_received,
            "involvement_count": n.involvement_count,
        }
        for n in network.nodes.values()
    ]

    with open(n_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader()
        writer.writerows(node_rows)

    # 2. Export Edges CSV
    edge_fields = [
        "team_id",
        "from_player_id",
        "to_player_id",
        "pass_count",
        "avg_transition_seconds",
        "avg_confidence",
    ]

    edge_rows = [
        {
            "team_id": e.team_id,
            "from_player_id": e.from_player_id,
            "to_player_id": e.to_player_id,
            "pass_count": e.pass_count,
            "avg_transition_seconds": e.avg_transition_seconds,
            "avg_confidence": e.avg_confidence,
        }
        for e in network.edges
    ]

    with open(e_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edge_rows)

    logger.info(
        f"Exported {len(node_rows)} nodes to {n_path} and {len(edge_rows)} edges to {e_path}"
    )
    return n_path, e_path


class PassNetworkGenerator:
    """
    Coordinator service for generating team pass networks, figures, and CSV datasets.
    """

    def __init__(
        self,
        config: AppConfig | PassNetworkConfig | None = None,
        output_dir: str | Path = "outputs/report",
    ):
        if isinstance(config, AppConfig):
            self.network_cfg = config.analytics.pass_network
        elif isinstance(config, PassNetworkConfig):
            self.network_cfg = config
        else:
            self.network_cfg = PassNetworkConfig()

        self.output_dir = Path(output_dir)
        self.networks_dir = self.output_dir / "pass_networks"
        self.data_dir = self.output_dir / "data"

    def load_events_from_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Load match candidate events from JSON or CSV."""
        p = Path(file_path)
        if not p.is_file():
            raise AnalyticsError(f"Events file not found: {file_path}")

        if p.suffix.lower() == ".json":
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                raise AnalyticsError(f"Expected list in events JSON, got {type(data)}")
            except json.JSONDecodeError as exc:
                raise AnalyticsError(f"Invalid JSON file {p}: {exc}") from exc

        elif p.suffix.lower() == ".csv":
            events: list[dict[str, Any]] = []
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    events.append(row)
            return events

        raise AnalyticsError(
            f"Unsupported event format '{p.suffix}'. Expected .json or .csv"
        )

    def generate(
        self,
        events: list[CandidateEvent | dict[str, Any]],
        intervals: list[Any] | None = None,
        player_records: list[dict[str, Any]] | None = None,
        team_filter: int | None = None,
    ) -> dict[str, Path]:
        """
        Generate pass network graphs, figures, and CSV datasets for teams.

        Returns:
            Dictionary mapping artifact names to their paths.
        """
        artifacts: dict[str, Path] = {}
        self.networks_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.network_cfg
        teams = [team_filter] if team_filter is not None else [1, 2]

        all_nodes_rows: list[dict[str, Any]] = []
        all_edges_rows: list[dict[str, Any]] = []

        for tid in teams:
            net = build_pass_network(
                events=events,
                intervals=intervals,
                player_records=player_records,
                team_id=tid,
                min_passes=cfg.min_passes,
                min_confidence=cfg.min_confidence,
            )

            out_img = self.networks_dir / f"team_{tid}_pass_network.png"
            render_pass_network(
                network=net,
                output_path=out_img,
                theme=cfg.theme,
                node_scale=cfg.node_scale,
                edge_scale=cfg.edge_scale,
            )
            plt.close("all")
            artifacts[f"team_{tid}_network"] = out_img

            # Accumulate CSV rows
            for n in net.nodes.values():
                all_nodes_rows.append(
                    {
                        "team_id": n.team_id,
                        "player_id": n.player_id,
                        "x_pitch_centroid": n.x_centroid,
                        "y_pitch_centroid": n.y_centroid,
                        "possession_seconds": n.possession_seconds,
                        "passes_sent": n.passes_sent,
                        "passes_received": n.passes_received,
                        "involvement_count": n.involvement_count,
                    }
                )

            for e in net.edges:
                all_edges_rows.append(
                    {
                        "team_id": e.team_id,
                        "from_player_id": e.from_player_id,
                        "to_player_id": e.to_player_id,
                        "pass_count": e.pass_count,
                        "avg_transition_seconds": e.avg_transition_seconds,
                        "avg_confidence": e.avg_confidence,
                    }
                )

        # Write consolidated nodes and edges CSV
        nodes_csv = self.data_dir / "pass_network_nodes.csv"
        edges_csv = self.data_dir / "pass_network_edges.csv"

        node_fields = [
            "team_id",
            "player_id",
            "x_pitch_centroid",
            "y_pitch_centroid",
            "possession_seconds",
            "passes_sent",
            "passes_received",
            "involvement_count",
        ]
        with open(nodes_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=node_fields)
            w.writeheader()
            w.writerows(all_nodes_rows)

        edge_fields = [
            "team_id",
            "from_player_id",
            "to_player_id",
            "pass_count",
            "avg_transition_seconds",
            "avg_confidence",
        ]
        with open(edges_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=edge_fields)
            w.writeheader()
            w.writerows(all_edges_rows)

        artifacts["pass_network_nodes_csv"] = nodes_csv
        artifacts["pass_network_edges_csv"] = edges_csv

        return artifacts

    def generate_from_files(
        self,
        events_path: str | Path,
        intervals_path: str | Path | None = None,
        tracking_path: str | Path | None = None,
        team_filter: int | None = None,
    ) -> dict[str, Path]:
        """Convenience method to generate pass networks directly from files."""
        events = self.load_events_from_file(events_path)

        player_records = None
        if tracking_path and Path(tracking_path).is_file():
            tp = Path(tracking_path)
            if tp.suffix.lower() == ".json":
                with open(tp, encoding="utf-8") as f:
                    player_records = json.load(f)
            elif tp.suffix.lower() == ".csv":
                player_records = []
                with open(tp, encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        player_records.append(row)

        return self.generate(
            events=events,
            player_records=player_records,
            team_filter=team_filter,
        )
