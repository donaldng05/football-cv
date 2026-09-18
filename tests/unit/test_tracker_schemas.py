"""
Unit tests for canonical tracking schemas and data structures.
"""

import pytest

from football_cv.tracking.schemas import MatchTracks, TrackedEntity


class TestTrackerSchemas:
    def test_tracked_entity_serialization(self):
        entity = TrackedEntity(
            bbox=[100.0, 200.0, 150.0, 300.0],
            position=(125, 300),
            speed=22.5,
            team=1,
            has_ball=True,
        )
        data = entity.to_dict()
        assert data["bbox"] == [100.0, 200.0, 150.0, 300.0]
        assert data["position"] == (125, 300)
        assert data["speed"] == 22.5
        assert data["team"] == 1
        assert data["has_ball"] is True

    def test_match_tracks_container(self):
        tracks = MatchTracks(
            players=[{1: {"bbox": [0, 0, 10, 10]}}],
            referees=[{100: {"bbox": [50, 50, 60, 60]}}],
            balls=[{1: {"bbox": [5, 5, 15, 15]}}],
        )

        # Dictionary-style indexing
        assert len(tracks["players"]) == 1
        assert len(tracks["referees"]) == 1
        assert len(tracks["balls"]) == 1

        # Mutation
        tracks["players"].append({2: {"bbox": [20, 20, 30, 30]}})
        assert len(tracks["players"]) == 2

        # Invalidation
        with pytest.raises(KeyError, match="Invalid track entity key"):
            _ = tracks["invalid_key"]

        with pytest.raises(KeyError, match="Invalid track entity key"):
            tracks["invalid_key"] = []

        # to_dict / from_dict
        d = tracks.to_dict()
        assert "players" in d
        restored = MatchTracks.from_dict(d)
        assert len(restored["players"]) == 2
