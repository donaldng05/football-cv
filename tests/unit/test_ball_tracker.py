"""
Unit tests for kinematic BallTracker.
"""

from football_cv.tracking.ball_tracker import BallTracker


class TestBallTracker:
    def test_ball_tracker_initialization(self):
        tracker = BallTracker(max_displacement_pixels=200.0, min_confidence=0.15)
        assert tracker.position is None
        assert tracker.velocity == (0.0, 0.0)
        assert tracker.lost_frames == 0

    def test_first_detection_initializes_tracker(self):
        tracker = BallTracker()
        candidates = [
            {"bbox": [100.0, 100.0, 120.0, 120.0], "confidence": 0.80},
            {"bbox": [500.0, 500.0, 520.0, 520.0], "confidence": 0.40},
        ]
        result = tracker.update(candidates)
        assert result is not None
        assert result["bbox"] == [100.0, 100.0, 120.0, 120.0]
        assert tracker.position == (110.0, 110.0)
        assert tracker.velocity == (0.0, 0.0)

    def test_low_confidence_candidate_ignored(self):
        tracker = BallTracker(min_confidence=0.50)
        candidates = [{"bbox": [100.0, 100.0, 120.0, 120.0], "confidence": 0.30}]
        result = tracker.update(candidates)
        assert result is None
        assert tracker.position is None

    def test_teleportation_false_positive_rejected(self):
        tracker = BallTracker(max_displacement_pixels=150.0)
        # Frame 1: Initial ball detection at (100, 100) -> center (110, 110)
        tracker.update([{"bbox": [100.0, 100.0, 120.0, 120.0], "confidence": 0.90}])

        # Frame 2: False positive at (600, 600) (distance ~700px > 150px)
        teleport_cand = [{"bbox": [600.0, 600.0, 620.0, 620.0], "confidence": 0.95}]
        result = tracker.update(teleport_cand)

        # Result should NOT follow the teleportation candidate to (600, 600)
        # Instead, it coasts near (110, 110)
        assert result is not None
        cx = (result["bbox"][0] + result["bbox"][2]) / 2.0
        cy = (result["bbox"][1] + result["bbox"][3]) / 2.0
        assert abs(cx - 110.0) < 10.0
        assert abs(cy - 110.0) < 10.0

    def test_scoring_prioritizes_proximity_over_marginal_confidence(self):
        tracker = BallTracker(max_displacement_pixels=200.0)
        # Initial position at (100, 100)
        tracker.update([{"bbox": [90.0, 90.0, 110.0, 110.0], "confidence": 0.90}])

        # Next frame: Candidate A is close (center 105, 105), conf = 0.85
        # Candidate B is far (center 260, 260, dist ~226px out of range or ~180px), conf = 0.95
        # Candidate A center = (105, 105), dist from (100, 100) = ~7px -> score ≈ 0.85 - 0.35*(7/200) ≈ 0.838
        # Candidate B center = (230, 230), dist = ~183px -> score ≈ 0.95 - 0.35*(183/200) ≈ 0.630
        candidates = [
            {"bbox": [220.0, 220.0, 240.0, 240.0], "confidence": 0.95},
            {"bbox": [95.0, 95.0, 115.0, 115.0], "confidence": 0.85},
        ]
        result = tracker.update(candidates)
        assert result is not None
        assert result["bbox"] == [95.0, 95.0, 115.0, 115.0]

    def test_coasting_during_brief_dropout(self):
        tracker = BallTracker()
        # Frame 0: at (100, 100) -> center (110, 110)
        tracker.update([{"bbox": [100.0, 100.0, 120.0, 120.0], "confidence": 0.90}])
        # Frame 1: moves to (110, 110) -> center (120, 120), velocity ≈ (6.0, 6.0)
        tracker.update([{"bbox": [110.0, 110.0, 130.0, 130.0], "confidence": 0.90}])

        # Frame 2: Dropout (no candidates) -> should coast using velocity
        coasted_frame = tracker.update([])
        assert coasted_frame is not None
        cx = (coasted_frame["bbox"][0] + coasted_frame["bbox"][2]) / 2.0
        cy = (coasted_frame["bbox"][1] + coasted_frame["bbox"][3]) / 2.0
        assert cx > 120.0
        assert cy > 120.0
        assert tracker.lost_frames == 1

        # Frame 3: Dropout 2 -> coasts again
        coasted_frame_2 = tracker.update([])
        assert coasted_frame_2 is not None
        assert tracker.lost_frames == 2

        # Frame 4: Dropout 3 -> coasting limit (<= 2) exceeded, returns None
        coasted_frame_3 = tracker.update([])
        assert coasted_frame_3 is None
        assert tracker.lost_frames == 3

    def test_reset_clears_state(self):
        tracker = BallTracker()
        tracker.update([{"bbox": [100.0, 100.0, 120.0, 120.0], "confidence": 0.90}])
        assert tracker.position is not None
        tracker.reset()
        assert tracker.position is None
        assert tracker.velocity == (0.0, 0.0)
        assert tracker.lost_frames == 0
