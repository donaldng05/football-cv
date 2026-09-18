"""
Unit tests for pass-network inference, graph calculation, and rendering.
"""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from football_cv.analytics.event_builder import CandidateEvent, EventType
from football_cv.analytics.pass_network import (
    PassNetwork,
    PassNetworkGenerator,
    build_pass_network,
    export_pass_network_csv,
    render_pass_network,
)
from football_cv.exceptions import AnalyticsError


@pytest.fixture
def sample_candidate_events() -> list[dict[str, Any]]:
    """Generate representative candidate pass events for testing."""
    return [
        # Team 1 passes
        {
            "event_id": 1,
            "event_type": EventType.CANDIDATE_PASS.value,
            "from_player_id": 10,
            "to_player_id": 7,
            "from_team_id": 1,
            "to_team_id": 1,
            "start_x_pitch": 45.0,
            "start_y_pitch": 25.0,
            "end_x_pitch": 60.0,
            "end_y_pitch": 30.0,
            "start_time": 2.0,
            "end_time": 3.2,
            "confidence": 0.90,
        },
        {
            "event_id": 2,
            "event_type": EventType.CANDIDATE_PASS.value,
            "from_player_id": 10,
            "to_player_id": 7,
            "from_team_id": 1,
            "to_team_id": 1,
            "start_x_pitch": 48.0,
            "start_y_pitch": 24.0,
            "end_x_pitch": 62.0,
            "end_y_pitch": 32.0,
            "start_time": 5.0,
            "end_time": 6.1,
            "confidence": 0.85,
        },
        {
            "event_id": 3,
            "event_type": EventType.CANDIDATE_PASS.value,
            "from_player_id": 7,
            "to_player_id": 9,
            "from_team_id": 1,
            "to_team_id": 1,
            "start_x_pitch": 61.0,
            "start_y_pitch": 31.0,
            "end_x_pitch": 80.0,
            "end_y_pitch": 34.0,
            "start_time": 6.5,
            "end_time": 7.8,
            "confidence": 0.88,
        },
        # Team 2 passes
        {
            "event_id": 4,
            "event_type": EventType.CANDIDATE_PASS.value,
            "from_player_id": 4,
            "to_player_id": 8,
            "from_team_id": 2,
            "to_team_id": 2,
            "start_x_pitch": 30.0,
            "start_y_pitch": 40.0,
            "end_x_pitch": 50.0,
            "end_y_pitch": 42.0,
            "start_time": 10.0,
            "end_time": 11.0,
            "confidence": 0.92,
        },
        # Non-pass events (should be filtered out)
        {
            "event_id": 5,
            "event_type": EventType.TURNOVER.value,
            "from_player_id": 9,
            "to_player_id": 4,
            "from_team_id": 1,
            "to_team_id": 2,
            "start_x_pitch": 82.0,
            "start_y_pitch": 35.0,
            "end_x_pitch": 80.0,
            "end_y_pitch": 36.0,
            "start_time": 8.0,
            "end_time": 8.5,
            "confidence": 0.80,
        },
        {
            "event_id": 6,
            "event_type": EventType.RECOVERY.value,
            "from_player_id": 4,
            "to_player_id": 4,
            "from_team_id": 2,
            "to_team_id": 2,
            "start_x_pitch": 32.0,
            "start_y_pitch": 41.0,
            "end_x_pitch": 32.0,
            "end_y_pitch": 41.0,
            "start_time": 12.0,
            "end_time": 12.5,
            "confidence": 0.95,
        },
    ]


class TestPassNetworkBuilding:
    """Test suite for build_pass_network graph construction."""

    def test_build_team_1_network(
        self, sample_candidate_events: list[dict[str, Any]]
    ) -> None:
        net = build_pass_network(sample_candidate_events, team_id=1, min_passes=1)
        assert isinstance(net, PassNetwork)
        assert net.team_id == 1
        assert net.total_passes == 3

        # Check nodes: 10, 7, 9
        assert set(net.nodes.keys()) == {10, 7, 9}
        node10 = net.nodes[10]
        assert node10.passes_sent == 2
        assert node10.passes_received == 0
        assert 45.0 <= node10.x_centroid <= 50.0

        node7 = net.nodes[7]
        assert node7.passes_received == 2
        assert node7.passes_sent == 1
        assert node7.involvement_count == 3

        # Check edges
        assert len(net.edges) == 2
        edge10_7 = next(
            e for e in net.edges if e.from_player_id == 10 and e.to_player_id == 7
        )
        assert edge10_7.pass_count == 2
        assert pytest.approx(edge10_7.avg_confidence, 0.01) == 0.875

    def test_min_passes_filtering(
        self, sample_candidate_events: list[dict[str, Any]]
    ) -> None:
        # Team 1 with min_passes=2 should only retain 10 -> 7
        net = build_pass_network(sample_candidate_events, team_id=1, min_passes=2)
        assert len(net.edges) == 1
        assert net.edges[0].from_player_id == 10
        assert net.edges[0].to_player_id == 7
        assert net.edges[0].pass_count == 2

    def test_min_confidence_filtering(
        self, sample_candidate_events: list[dict[str, Any]]
    ) -> None:
        net = build_pass_network(
            sample_candidate_events, team_id=1, min_confidence=0.89
        )
        assert net.total_passes == 1  # Only event_id=1 has confidence >= 0.89

    def test_candidate_event_objects_supported(self) -> None:
        ev = CandidateEvent(
            event_id=1,
            start_frame=0,
            end_frame=10,
            start_time=0.0,
            end_time=0.4,
            from_player_id=10,
            to_player_id=7,
            from_team_id=1,
            to_team_id=1,
            start_x_pitch=50.0,
            start_y_pitch=30.0,
            end_x_pitch=60.0,
            end_y_pitch=35.0,
            start_x_image=100.0,
            start_y_image=100.0,
            end_x_image=200.0,
            end_y_image=200.0,
            transition_frames=10,
            event_type=EventType.CANDIDATE_PASS.value,
            confidence=0.9,
        )
        net = build_pass_network([ev], team_id=1)
        assert net.total_passes == 1
        assert 10 in net.nodes
        assert 7 in net.nodes

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(AnalyticsError, match="min_passes must be >= 1"):
            build_pass_network([], min_passes=0)
        with pytest.raises(AnalyticsError, match="Pitch dimensions must be > 0"):
            build_pass_network([], pitch_length=-10.0)


class TestPassNetworkRenderingAndExport:
    """Test suite for figure rendering and CSV dataset export."""

    def test_render_pass_network(
        self, sample_candidate_events: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        net = build_pass_network(sample_candidate_events, team_id=1)
        out_png = tmp_path / "team_1_network.png"

        fig, _ = render_pass_network(
            network=net,
            output_path=out_png,
            theme="tactical_dark",
        )
        assert out_png.is_file()
        assert out_png.stat().st_size > 1000
        plt.close(fig)

    def test_export_pass_network_csv(
        self, sample_candidate_events: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        net = build_pass_network(sample_candidate_events, team_id=1)
        nodes_csv = tmp_path / "nodes.csv"
        edges_csv = tmp_path / "edges.csv"

        n_p, e_p = export_pass_network_csv(net, nodes_csv, edges_csv)
        assert n_p.is_file()
        assert e_p.is_file()

        with open(nodes_csv, encoding="utf-8") as f:
            n_rows = list(csv.DictReader(f))
            assert len(n_rows) == 3
            assert "x_pitch_centroid" in n_rows[0]
            assert "passes_sent" in n_rows[0]

        with open(edges_csv, encoding="utf-8") as f:
            e_rows = list(csv.DictReader(f))
            assert len(e_rows) == 2
            assert "from_player_id" in e_rows[0]
            assert "pass_count" in e_rows[0]


class TestPassNetworkGenerator:
    """Test suite for the PassNetworkGenerator coordinator."""

    def test_generate_from_events(
        self, sample_candidate_events: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        gen = PassNetworkGenerator(output_dir=tmp_path)
        artifacts = gen.generate(sample_candidate_events)

        assert "team_1_network" in artifacts
        assert "team_2_network" in artifacts
        assert "pass_network_nodes_csv" in artifacts
        assert "pass_network_edges_csv" in artifacts

        assert artifacts["team_1_network"].is_file()
        assert artifacts["team_2_network"].is_file()
        assert artifacts["pass_network_nodes_csv"].is_file()
        assert artifacts["pass_network_edges_csv"].is_file()

    def test_generate_from_files(
        self, sample_candidate_events: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        json_file = tmp_path / "events.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_candidate_events, f)

        gen = PassNetworkGenerator(output_dir=tmp_path / "report")
        artifacts = gen.generate_from_files(json_file)
        assert len(artifacts) >= 4

    def test_load_events_invalid_file(self, tmp_path: Path) -> None:
        gen = PassNetworkGenerator(output_dir=tmp_path)
        with pytest.raises(AnalyticsError, match="not found"):
            gen.load_events_from_file(tmp_path / "missing.json")

        bad = tmp_path / "bad.txt"
        bad.write_text("hello", encoding="utf-8")
        with pytest.raises(AnalyticsError, match="Unsupported event format"):
            gen.load_events_from_file(bad)
