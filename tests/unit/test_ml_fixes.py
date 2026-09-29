"""
Comprehensive unit test suite for ML & Tracking fixes:
1. Class-specific confidence filtering (high recall for ball, noise suppression for players).
2. Goalkeeper role preservation and KMeans centroid isolation.
3. Spatial goalmouth team attribution for goalkeepers.
4. Trajectory continuity (no 1-frame teleportation spikes) in TrackSanitizer.
5. Appearance feature extraction and appearance-gated tracklet stitching.
"""

import numpy as np
import pytest
import supervision as sv

from football_cv.teams.classifier import TeamClassifier
from football_cv.tracking.detector import ObjectDetector
from football_cv.tracking.reid import TrackSanitizer


class TestClassSpecificConfidence:
    """Test suite for class-specific confidence thresholding in ObjectDetector."""

    def test_filter_by_class_confidence_sv_detections(self):
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.confidence = 0.25
        detector.names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        detector.names_inv = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
        detector.class_confidences = {
            "player": 0.25,
            "goalkeeper": 0.25,
            "referee": 0.25,
            "ball": 0.12,
        }

        # Detections:
        # 1. Player with conf 0.18 (below 0.25 -> should be rejected)
        # 2. Player with conf 0.28 (above 0.25 -> should be kept)
        # 3. Ball with conf 0.14 (above 0.12 -> should be kept)
        # 4. Ball with conf 0.08 (below 0.12 -> should be rejected)
        # 5. Referee with conf 0.20 (below 0.25 -> should be rejected)
        raw_dets = sv.Detections(
            xyxy=np.array(
                [
                    [100, 100, 150, 200],
                    [200, 200, 250, 300],
                    [300, 300, 310, 310],
                    [400, 400, 410, 410],
                    [500, 500, 550, 600],
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.18, 0.28, 0.14, 0.08, 0.20], dtype=np.float32),
            class_id=np.array([2, 2, 0, 0, 3], dtype=int),
        )

        filtered = detector._filter_by_class_confidence(raw_dets)
        assert len(filtered) == 2
        # Only Player 2 (conf 0.28) and Ball (conf 0.14) remain
        assert filtered.class_id.tolist() == [2, 0]
        np.testing.assert_allclose(filtered.confidence, [0.28, 0.14], atol=1e-5)

    def test_effective_conf_computation(self):
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.confidence = 0.30
        detector.class_confidences = {"player": 0.30, "ball": 0.12}
        assert detector.effective_conf == 0.12

        detector.class_confidences = None
        assert detector.effective_conf == 0.30


class TestGoalkeeperIsolationAndAttribution:
    """Test suite ensuring goalkeepers do not poison KMeans and are assigned teams spatially."""

    def test_goalkeeper_excluded_from_kmeans_sampling(self):
        classifier = TeamClassifier(color_space="bgr")

        # Frame with 2 outfield players (Team 1: Red, Team 2: Blue)
        # and 1 Goalkeeper with fluorescent neon yellow jersey (0, 255, 255)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Player 1 (Red): (0, 0, 255)
        frame[100:150, 100:150] = (0, 0, 255)
        # Player 2 (Blue): (255, 0, 0)
        frame[100:150, 300:350] = (255, 0, 0)
        # Goalkeeper (Neon Yellow): (0, 255, 255)
        frame[100:150, 500:550] = (0, 255, 255)

        player_tracks = [
            {
                1: {"bbox": [100.0, 100.0, 150.0, 200.0], "role": "player"},
                2: {"bbox": [300.0, 100.0, 350.0, 200.0], "role": "player"},
                3: {"bbox": [500.0, 100.0, 550.0, 200.0], "role": "goalkeeper"},
            }
        ]

        classifier.assign_team_color(
            [frame], player_tracks, sample_frames=1, min_box_area=100.0
        )

        assert classifier.kmeans is not None
        # Centroids should be around Red and Blue, NOT corrupted by Neon Yellow
        c1, c2 = classifier.kmeans.cluster_centers_
        # Neither centroid should have high green (which neon yellow has: 255)
        assert c1[1] < 100.0 or c2[1] < 100.0

    def test_goalkeeper_pitch_spatial_attribution(self):
        classifier = TeamClassifier(color_space="bgr")
        classifier.kmeans = None  # Ensure it doesn't use color for GK

        # Team 1 is on the left (mean x ~ 300), Team 2 is on the right (mean x ~ 900)
        frame_players = {
            1: {"bbox": [250.0, 200.0, 290.0, 300.0], "team": 1, "role": "player"},
            2: {"bbox": [310.0, 300.0, 350.0, 400.0], "team": 1, "role": "player"},
            3: {"bbox": [850.0, 200.0, 890.0, 300.0], "team": 2, "role": "player"},
            4: {"bbox": [910.0, 300.0, 950.0, 400.0], "team": 2, "role": "player"},
        }

        # GK 10 is at x=100 (left goalmouth) -> defends left side -> belongs to Team 1
        gk1_bbox = [80.0, 300.0, 120.0, 400.0]
        team_gk1 = classifier._assign_goalkeeper_team(gk1_bbox, frame_players)
        assert team_gk1 == 1

        # GK 20 is at x=1100 (right goalmouth) -> defends right side -> belongs to Team 2
        gk2_bbox = [1080.0, 300.0, 1120.0, 400.0]
        team_gk2 = classifier._assign_goalkeeper_team(gk2_bbox, frame_players)
        assert team_gk2 == 2


class TestTrajectoryContinuityAndReID:
    """Test suite ensuring ID switches maintain kinematic continuity without 1-frame teleportation."""

    def test_no_teleportation_on_isolated_classification_noise(self):
        # Player 1 is physically moving from x=100 to x=190
        # Frame 5 has a classification noise (team=2 instead of 1)
        # Player 2 is physically moving from x=500 to x=590
        # Frame 5 has a classification noise (team=1 instead of 2)
        player_tracks = []
        for f in range(10):
            p1_team = 2 if f == 5 else 1
            p2_team = 1 if f == 5 else 2
            player_tracks.append(
                {
                    1: {
                        "bbox": [100.0 + f * 10, 200.0, 140.0 + f * 10, 300.0],
                        "team": p1_team,
                    },
                    2: {
                        "bbox": [500.0 + f * 10, 200.0, 540.0 + f * 10, 300.0],
                        "team": p2_team,
                    },
                }
            )

        TrackSanitizer.sanitize_team_consistency(
            player_tracks, max_crossover_distance=100.0, max_swap_velocity_px=150.0
        )

        # Assert teams were cleaned
        for f in range(10):
            assert player_tracks[f][1]["team"] == 1
            assert player_tracks[f][2]["team"] == 2

        # CRITICAL: Verify NO 1-frame position teleportation spike occurred at frame 5!
        for f in range(1, 10):
            prev_x = (
                player_tracks[f - 1][1]["bbox"][0] + player_tracks[f - 1][1]["bbox"][2]
            ) / 2.0
            cur_x = (
                player_tracks[f][1]["bbox"][0] + player_tracks[f][1]["bbox"][2]
            ) / 2.0
            displacement = abs(cur_x - prev_x)
            assert displacement <= 15.0, (
                f"Teleportation spike at frame {f}: {displacement} px"
            )

    def test_appearance_feature_extraction_and_distance(self):
        # Create two distinct color crops: Blue torso vs Red torso
        frame_blue = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_blue[:50, :] = (255, 0, 0)  # Blue upper torso

        frame_red = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_red[:50, :] = (0, 0, 255)  # Red upper torso

        feat_blue = TrackSanitizer.extract_appearance_features(
            frame_blue, [0, 0, 100, 100]
        )
        feat_red = TrackSanitizer.extract_appearance_features(
            frame_red, [0, 0, 100, 100]
        )

        assert feat_blue.shape == (48,)
        assert feat_red.shape == (48,)
        np.testing.assert_allclose(np.linalg.norm(feat_blue), 1.0, atol=1e-5)

        # Distance to identical feature should be 0.0
        dist_same = TrackSanitizer.compute_appearance_distance(feat_blue, feat_blue)
        assert dist_same == pytest.approx(0.0, abs=1e-4)

        # Distance between Blue and Red should be high (> 0.50)
        dist_diff = TrackSanitizer.compute_appearance_distance(feat_blue, feat_red)
        assert dist_diff > 0.50

    def test_appearance_gated_stitching_rejects_dissimilar_players(self):
        # Track 1 ends frame 4 (Blue player)
        # Track 2 starts frame 6 (Red player) in same location
        player_tracks = []
        for f in range(10):
            frame_dict = {}
            if 0 <= f <= 4:
                frame_dict[1] = {"bbox": [100.0, 200.0, 140.0, 300.0], "team": 1}
            elif 6 <= f <= 9:
                frame_dict[2] = {"bbox": [105.0, 205.0, 145.0, 305.0], "team": 1}
            player_tracks.append(frame_dict)

        # Create distinct appearance features for track 1 and track 2
        feat1 = np.zeros(48, dtype=np.float32)
        feat1[:24] = 1.0
        feat1 /= np.linalg.norm(feat1)

        feat2 = np.zeros(48, dtype=np.float32)
        feat2[24:] = 1.0
        feat2 /= np.linalg.norm(feat2)

        appearance_features = {1: feat1, 2: feat2}

        TrackSanitizer.stitch_fragmented_tracks(
            player_tracks,
            max_gap_frames=10,
            max_pixel_distance=50.0,
            max_appearance_distance=0.30,
            appearance_features=appearance_features,
        )

        # Track 2 should NOT be merged into Track 1 because appearance is dissimilar!
        for f in range(6, 10):
            assert 2 in player_tracks[f]
            assert 1 not in player_tracks[f]
