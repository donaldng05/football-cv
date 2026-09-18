"""
Football analytics and structured data export package.
"""

from .event_builder import CandidateEvent, EventBuilder, EventType
from .exporter import AnalyticsExporter

__all__ = [
    "AnalyticsExporter",
    "CandidateEvent",
    "EventBuilder",
    "EventType",
]
