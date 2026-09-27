"""
Unit tests for TeamClassifier (perceptual Lab clustering, multi-frame sampling, and temporal voting).
"""

import numpy as np

from football_cv.teams.classifier import TeamClassifier


class TestTeamClassifier:
    def test_team_classifier_initialization(self):
        classifier = TeamClassifier(color_space="lab", voting_window=5)
        assert classifier.color_space == "lab"
        assert classifier.voting_window == 5
        assert classifier.kmeans is None
        assert len(classifier.team_colors) == 0

    def test_single_frame_fallback(self):
        classifier = TeamClassifier(color_space="bgr")
        # Create a synthetic frame with two player crops: Team 1 (red), Team 2 (blue)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Red player at (10, 10, 50, 70)
        frame[10:70, 10:50] = [0, 0, 255]
        # Blue player at (100, 10, 140, 70)
        frame[10:70, 100:140] = [255, 0, 0]

        detections = {
            1: {"bbox": [10.0, 10.0, 50.0, 70.0]},
            2: {"bbox": [100.0, 10.0, 140.0, 70.0]},
        }
        classifier.assign_team_color(frame, detections)
        assert classifier.kmeans is not None
        assert 1 in classifier.team_colors
        assert 2 in classifier.team_colors

        # Query team predictions
        t1 = classifier.get_player_team(frame, [10.0, 10.0, 50.0, 70.0], player_id=1)
        t2 = classifier.get_player_team(frame, [100.0, 10.0, 140.0, 70.0], player_id=2)
        assert t1 != t2
        assert {t1, t2} == {1, 2}

    def test_multi_frame_jersey_sampling(self):
        classifier = TeamClassifier(color_space="lab", voting_window=5)
        frames = []
        player_tracks = []

        # Create 10 frames where Team 1 is green [0, 255, 0] and Team 2 is white [255, 255, 255]
        for _f in range(10):
            frame = np.zeros((400, 400, 3), dtype=np.uint8)
            # Add field background (dark green)
            frame[:, :] = [30, 100, 30]

            # Player 1 (bright green jersey): center of torso
            frame[20:100, 20:80] = [50, 240, 50]
            # Player 2 (white jersey): center of torso
            frame[20:100, 150:210] = [240, 240, 240]

            frames.append(frame)
            player_tracks.append(
                {
                    1: {"bbox": [20.0, 20.0, 80.0, 100.0]},
                    2: {"bbox": [150.0, 20.0, 210.0, 100.0]},
                }
            )

        classifier.assign_team_color(frames, player_tracks, sample_frames=5)
        assert classifier.kmeans is not None

        # Verify predictions separate players into distinct teams
        team_p1 = classifier.get_player_team(
            frames[0], [20.0, 20.0, 80.0, 100.0], player_id=1
        )
        team_p2 = classifier.get_player_team(
            frames[0], [150.0, 20.0, 210.0, 100.0], player_id=2
        )
        assert team_p1 != team_p2

    def test_temporal_majority_voting(self):
        classifier = TeamClassifier(color_space="bgr", voting_window=3)
        # Pre-seed KMeans model with known centers
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=2, random_state=42)
        # Cluster 0: Red [0, 0, 255], Cluster 1: Blue [255, 0, 0]
        samples = np.array([[0.0, 0.0, 255.0], [255.0, 0.0, 0.0]])
        kmeans.fit(samples)
        classifier.kmeans = kmeans

        # Frame 0: Red jersey (Team predicted as red)
        f_red = np.zeros((100, 100, 3), dtype=np.uint8)
        f_red[:, :] = [0, 0, 255]

        # Frame 1: Glitched frame with blue noise
        f_blue = np.zeros((100, 100, 3), dtype=np.uint8)
        f_blue[:, :] = [255, 0, 0]

        # Query 1 (vote 1: red)
        t_v1 = classifier.get_player_team(f_red, [10.0, 10.0, 80.0, 80.0], player_id=42)
        # Query 2 (vote 2: glitched blue)
        classifier.get_player_team(f_blue, [10.0, 10.0, 80.0, 80.0], player_id=42)
        # Query 3 (vote 3: red)
        t_v3 = classifier.get_player_team(f_red, [10.0, 10.0, 80.0, 80.0], player_id=42)

        # Majority vote across [red, blue, red] should be red
        assert t_v3 == t_v1
        assert classifier.player_team_dict[42] == t_v1
