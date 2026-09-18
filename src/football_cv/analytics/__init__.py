"""
Football analytics and structured data export package.
"""

from .error_analysis import (
    BENCHMARK_CASES,
    CaseEvaluationResult,
    ErrorAnalysisExporter,
    ErrorReport,
    FailureCase,
    FailureCaseEvaluator,
    FailureCategory,
    FailureMode,
)
from .event_builder import CandidateEvent, EventBuilder, EventType
from .exporter import AnalyticsExporter
from .heatmap import (
    HeatmapGenerator,
    PossessionGrid,
    aggregate_possession_grid,
    export_heatmap_csv,
    filter_possession_records,
    render_possession_heatmap,
)
from .pass_network import (
    PassEdge,
    PassNetwork,
    PassNetworkGenerator,
    PassNode,
    build_pass_network,
    export_pass_network_csv,
    render_pass_network,
)
from .pitch import create_pitch_figure, draw_pitch

__all__ = [
    "BENCHMARK_CASES",
    "AnalyticsExporter",
    "CandidateEvent",
    "CaseEvaluationResult",
    "ErrorAnalysisExporter",
    "ErrorReport",
    "EventBuilder",
    "EventType",
    "FailureCase",
    "FailureCaseEvaluator",
    "FailureCategory",
    "FailureMode",
    "HeatmapGenerator",
    "PassEdge",
    "PassNetwork",
    "PassNetworkGenerator",
    "PassNode",
    "PossessionGrid",
    "aggregate_possession_grid",
    "build_pass_network",
    "create_pitch_figure",
    "draw_pitch",
    "export_heatmap_csv",
    "export_pass_network_csv",
    "filter_possession_records",
    "render_pass_network",
    "render_possession_heatmap",
]
