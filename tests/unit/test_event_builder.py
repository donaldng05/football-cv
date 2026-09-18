"""
Unit tests for candidate tactical match event inference.
"""

from football_cv.analytics.event_builder import (
    CandidateEvent,
    EventBuilder,
    EventType,
)
from football_cv.possession.events import PossessionInterval


class TestEventBuilder:
    def test_build_candidate_pass_teammates(self):
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=5,
                start_time=0.0,
                end_time=0.2,
                player_id=1,
                team_id=1,
                duration_seconds=0.24,
                frame_count=6,
                termination_reason="pass",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=8,
                end_frame=15,
                start_time=0.32,
                end_time=0.6,
                player_id=2,
                team_id=1,
                duration_seconds=0.32,
                frame_count=8,
                termination_reason="end_of_clip",
            ),
        ]
        player_tracks = [
            {
                1: {
                    "position_transformed": [25.0, 15.0],
                    "position_adjusted": (200.0, 300.0),
                },
                2: {
                    "position_transformed": [28.0, 16.0],
                    "position_adjusted": (500.0, 400.0),
                },
            }
            for _ in range(16)
        ]

        builder = EventBuilder(maximum_transition_frames=10, fps=25.0)
        events = builder.build_events(intervals, player_tracks)

        assert len(events) == 1
        ev = events[0]
        assert ev.event_id == 1
        assert ev.event_type == EventType.CANDIDATE_PASS.value
        assert ev.from_player_id == 1
        assert ev.to_player_id == 2
        assert ev.from_team_id == 1
        assert ev.to_team_id == 1
        assert ev.transition_frames == 3
        assert ev.start_x_pitch == 25.0
        assert ev.start_y_pitch == 15.0
        assert ev.end_x_pitch == 28.0
        assert ev.end_y_pitch == 16.0
        assert ev.confidence >= 0.8

    def test_build_turnover_opposing_teams(self):
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=4,
                start_time=0.0,
                end_time=0.16,
                player_id=1,
                team_id=1,
                duration_seconds=0.2,
                frame_count=5,
                termination_reason="turnover",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=7,
                end_frame=12,
                start_time=0.28,
                end_time=0.48,
                player_id=10,
                team_id=2,
                duration_seconds=0.24,
                frame_count=6,
                termination_reason="end_of_clip",
            ),
        ]
        player_tracks = [
            {
                1: {
                    "position_transformed": [30.0, 15.0],
                    "position_adjusted": (300.0, 200.0),
                },
                10: {
                    "position_transformed": [35.0, 18.0],
                    "position_adjusted": (350.0, 250.0),
                },
            }
            for _ in range(15)
        ]

        builder = EventBuilder(maximum_transition_frames=10, fps=25.0)
        events = builder.build_events(intervals, player_tracks)

        assert len(events) == 1
        assert events[0].event_type == EventType.TURNOVER.value
        assert events[0].from_team_id == 1
        assert events[0].to_team_id == 2

    def test_build_recovery_same_player(self):
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=3,
                start_time=0.0,
                end_time=0.12,
                player_id=1,
                team_id=1,
                duration_seconds=0.16,
                frame_count=4,
                termination_reason="lost_ball",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=5,
                end_frame=9,
                start_time=0.2,
                end_time=0.36,
                player_id=1,
                team_id=1,
                duration_seconds=0.2,
                frame_count=5,
                termination_reason="end_of_clip",
            ),
        ]
        player_tracks = [
            {
                1: {
                    "position_transformed": [20.0, 10.0],
                    "position_adjusted": (150.0, 120.0),
                }
            }
            for _ in range(10)
        ]

        builder = EventBuilder(maximum_transition_frames=10, fps=25.0)
        events = builder.build_events(intervals, player_tracks)

        assert len(events) == 1
        assert events[0].event_type == EventType.RECOVERY.value
        assert events[0].from_player_id == 1
        assert events[0].to_player_id == 1

    def test_build_uncertain_transition_when_uncalibrated(self):
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=2,
                start_time=0.0,
                end_time=0.08,
                player_id=1,
                team_id=1,
                duration_seconds=0.12,
                frame_count=3,
                termination_reason="pass",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=4,
                end_frame=6,
                start_time=0.16,
                end_time=0.24,
                player_id=2,
                team_id=1,
                duration_seconds=0.12,
                frame_count=3,
                termination_reason="end_of_clip",
            ),
        ]
        # Position transformed is None for players
        player_tracks = [
            {1: {"position": (100, 100)}, 2: {"position": (200, 200)}}
            for _ in range(10)
        ]

        builder = EventBuilder(maximum_transition_frames=10, fps=25.0)
        events = builder.build_events(intervals, player_tracks)

        assert len(events) == 1
        assert events[0].event_type == EventType.UNCERTAIN_TRANSITION.value

    def test_build_events_empty_or_single_interval(self):
        builder = EventBuilder()
        assert builder.build_events([], []) == []
        assert (
            builder.build_events(
                [
                    PossessionInterval(
                        interval_id=1,
                        start_frame=0,
                        end_frame=2,
                        start_time=0.0,
                        end_time=0.08,
                        player_id=1,
                        team_id=1,
                        duration_seconds=0.12,
                        frame_count=3,
                        termination_reason="end_of_clip",
                    )
                ],
                [],
            )
            == []
        )

    def test_candidate_event_to_dict(self):
        ev = CandidateEvent(
            event_id=1,
            start_frame=0,
            end_frame=5,
            start_time=0.0,
            end_time=0.2,
            from_player_id=1,
            to_player_id=2,
            from_team_id=1,
            to_team_id=1,
            start_x_pitch=10.0,
            start_y_pitch=15.0,
            end_x_pitch=25.0,
            end_y_pitch=30.0,
            start_x_image=100.0,
            start_y_image=150.0,
            end_x_image=250.0,
            end_y_image=300.0,
            transition_frames=2,
            event_type="candidate_pass",
            confidence=0.88,
        )
        d = ev.to_dict()
        assert d["event_id"] == 1
        assert d["event_type"] == "candidate_pass"
        assert d["transition_frames"] == 2
        assert d["confidence"] == 0.88

    def test_reject_track_swap_identity_switch(self):
        # Two tracks on the same team, separated by 0.5m in 1 frame (identity switch)
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=4,
                start_time=0.0,
                end_time=0.16,
                player_id=1,
                team_id=1,
                duration_seconds=0.2,
                frame_count=5,
                termination_reason="pass",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=5,
                end_frame=10,
                start_time=0.2,
                end_time=0.4,
                player_id=2,
                team_id=1,
                duration_seconds=0.24,
                frame_count=6,
                termination_reason="end_of_clip",
            ),
        ]
        player_tracks = [
            {
                1: {"position_transformed": [30.0, 20.0]},
                2: {"position_transformed": [30.3, 20.4]},  # dist = 0.5m
            }
            for _ in range(12)
        ]

        # With kinematics validation enabled (default)
        builder = EventBuilder(fps=25.0)
        events = builder.build_events(intervals, player_tracks)
        assert len(events) == 1
        assert events[0].event_type == EventType.EXCLUDED.value
        assert events[0].confidence == 0.95

        # With kinematics validation disabled, it is misclassified as a pass
        builder_disabled = EventBuilder(fps=25.0, validate_kinematics=False)
        events_disabled = builder_disabled.build_events(intervals, player_tracks)
        assert len(events_disabled) == 1
        assert events_disabled[0].event_type == EventType.CANDIDATE_PASS.value

    def test_reject_superhuman_pass_velocity(self):
        # Two tracks separated by 60m in 2 frames (v = 750 m/s > 45 m/s)
        intervals = [
            PossessionInterval(
                interval_id=1,
                start_frame=0,
                end_frame=3,
                start_time=0.0,
                end_time=0.12,
                player_id=1,
                team_id=1,
                duration_seconds=0.16,
                frame_count=4,
                termination_reason="pass",
            ),
            PossessionInterval(
                interval_id=2,
                start_frame=5,
                end_frame=10,
                start_time=0.2,
                end_time=0.4,
                player_id=2,
                team_id=1,
                duration_seconds=0.24,
                frame_count=6,
                termination_reason="end_of_clip",
            ),
        ]
        player_tracks = [
            {
                1: {"position_transformed": [10.0, 10.0]},
                2: {"position_transformed": [70.0, 10.0]},  # dist = 60m in 2 frames
            }
            for _ in range(12)
        ]

        builder = EventBuilder(fps=25.0, maximum_pass_speed=45.0)
        events = builder.build_events(intervals, player_tracks)
        assert len(events) == 1
        assert events[0].event_type == EventType.UNCERTAIN_TRANSITION.value
        assert events[0].confidence == 0.40
