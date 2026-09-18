"""
Unit tests for possession interval extraction and filtering logic.
"""

from football_cv.possession.events import (
    PossessionInterval,
    PossessionIntervalExtractor,
)


class TestPossessionIntervalExtractor:
    def test_extract_intervals_continuous_possession(self):
        # 5 frames where Player 1 holds the ball continuously
        player_tracks = [
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
        ]
        extractor = PossessionIntervalExtractor(minimum_control_frames=2, fps=25.0)
        intervals = extractor.extract_intervals(player_tracks)

        assert len(intervals) == 1
        interval = intervals[0]
        assert interval.interval_id == 1
        assert interval.player_id == 1
        assert interval.team_id == 1
        assert interval.start_frame == 0
        assert interval.end_frame == 4
        assert interval.frame_count == 5
        assert interval.duration_seconds == 0.2
        assert interval.termination_reason == "end_of_clip"

    def test_extract_intervals_filter_short_jitter(self):
        # Player 1 holds for 1 frame, Player 2 holds for 4 frames
        player_tracks = [
            {1: {"has_ball": True, "team": 1}},  # 1 frame (below min threshold 2)
            {2: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 1}},
        ]
        extractor = PossessionIntervalExtractor(minimum_control_frames=2, fps=25.0)
        intervals = extractor.extract_intervals(player_tracks)

        # 1-frame jitter is filtered out
        assert len(intervals) == 1
        assert intervals[0].player_id == 2
        assert intervals[0].frame_count == 4

    def test_extract_intervals_handoff_and_termination_reasons(self):
        # Sequence:
        # Frames 0..2: Player 1 (Team 1) -> passes to Player 2 (Team 1)
        # Frames 3..5: Player 2 (Team 1) -> turnover to Player 10 (Team 2)
        # Frames 6..8: Player 10 (Team 2) -> ball lost
        # Frames 9: Nobody
        player_tracks = [
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {1: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 1}},
            {2: {"has_ball": True, "team": 1}},
            {10: {"has_ball": True, "team": 2}},
            {10: {"has_ball": True, "team": 2}},
            {10: {"has_ball": True, "team": 2}},
            {10: {"has_ball": False, "team": 2}},
        ]
        extractor = PossessionIntervalExtractor(minimum_control_frames=1, fps=25.0)
        intervals = extractor.extract_intervals(player_tracks)

        assert len(intervals) == 3
        # Interval 1: pass to teammate
        assert intervals[0].player_id == 1
        assert intervals[0].team_id == 1
        assert intervals[0].termination_reason == "pass"

        # Interval 2: turnover to opponent
        assert intervals[1].player_id == 2
        assert intervals[1].team_id == 1
        assert intervals[1].termination_reason == "turnover"

        # Interval 3: ball lost to loose state
        assert intervals[2].player_id == 10
        assert intervals[2].team_id == 2
        assert intervals[2].termination_reason == "lost_ball"

    def test_extract_intervals_empty_tracks(self):
        extractor = PossessionIntervalExtractor()
        assert extractor.extract_intervals([]) == []
        assert extractor.extract_intervals([{}, {}]) == []

    def test_possession_interval_to_dict(self):
        interval = PossessionInterval(
            interval_id=1,
            start_frame=0,
            end_frame=10,
            start_time=0.0,
            end_time=0.4,
            player_id=7,
            team_id=1,
            duration_seconds=0.44,
            frame_count=11,
            confidence=0.95,
            termination_reason="pass",
        )
        d = interval.to_dict()
        assert d["interval_id"] == 1
        assert d["player_id"] == 7
        assert d["duration_seconds"] == 0.44
        assert d["termination_reason"] == "pass"
