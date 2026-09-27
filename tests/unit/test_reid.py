"""
Unit tests for Re-ID TrackSanitizer.
"""

from football_cv.tracking.reid import TrackSanitizer


class TestTrackSanitizer:
    def test_sanitize_team_consistency_swaps_crossing_players(self):
        # 10 frames simulation:
        # Player 1: Team 1 in frames 0..4, gets inverted to Team 2 in frames 5..6, returns to Team 1 in frames 7..9
        # Player 2: Team 2 in frames 0..4, gets inverted to Team 1 in frames 5..6, returns to Team 2 in frames 7..9
        player_tracks = []
        for f in range(10):
            if f in (5, 6):
                # Symmetrical swap due to tracking crossover
                p1_team = 2
                p2_team = 1
            else:
                p1_team = 1
                p2_team = 2

            frame_dict = {
                1: {
                    "bbox": [100.0 + f * 5, 200.0, 140.0 + f * 5, 300.0],
                    "team": p1_team,
                },
                2: {
                    "bbox": [300.0 - f * 5, 200.0, 340.0 - f * 5, 300.0],
                    "team": p2_team,
                },
            }
            player_tracks.append(frame_dict)

        TrackSanitizer.sanitize_team_consistency(player_tracks)

        # In all frames, Player 1 should now be Team 1 and Player 2 should be Team 2
        for f in range(10):
            assert player_tracks[f][1]["team"] == 1, (
                f"Frame {f} Player 1 team incorrect"
            )
            assert player_tracks[f][2]["team"] == 2, (
                f"Frame {f} Player 2 team incorrect"
            )

    def test_single_frame_team_glitch_aligned_to_dominant(self):
        # Player 7 belongs to Team 1 across 10 frames, with a single frame glitch on frame 4
        player_tracks = []
        for f in range(10):
            team = 2 if f == 4 else 1
            player_tracks.append(
                {7: {"bbox": [100.0, 200.0, 140.0, 300.0], "team": team}}
            )

        TrackSanitizer.sanitize_team_consistency(player_tracks)
        for f in range(10):
            assert player_tracks[f][7]["team"] == 1

    def test_stitch_fragmented_tracks_merges_nearby_tracklets(self):
        # Track 1 active frames 0..4, ends at center (120, 250)
        # Track 2 appears frame 7 at center (125, 255) (gap 3 frames, distance ~7px <= 120px)
        player_tracks = []
        for f in range(12):
            frame_dict = {}
            if 0 <= f <= 4:
                frame_dict[1] = {
                    "bbox": [100.0 + f, 200.0, 140.0 + f, 300.0],
                    "team": 1,
                }
            elif 7 <= f <= 11:
                frame_dict[2] = {
                    "bbox": [105.0 + f, 205.0, 145.0 + f, 305.0],
                    "team": 1,
                }
            player_tracks.append(frame_dict)

        TrackSanitizer.stitch_fragmented_tracks(
            player_tracks, max_gap_frames=10, max_pixel_distance=120.0
        )

        # Track 2 should now be merged into Track 1 in frames 7..11
        for f in range(7, 12):
            assert 1 in player_tracks[f], f"Track 1 should be stitched in frame {f}"
            assert 2 not in player_tracks[f], f"Track 2 should be gone in frame {f}"

    def test_stitch_fragmented_tracks_rejects_different_teams(self):
        player_tracks = []
        for f in range(10):
            frame_dict = {}
            if 0 <= f <= 4:
                # Team 1 ends
                frame_dict[1] = {"bbox": [100.0, 200.0, 140.0, 300.0], "team": 1}
            elif 6 <= f <= 9:
                # Team 2 appears in same location
                frame_dict[2] = {"bbox": [100.0, 200.0, 140.0, 300.0], "team": 2}
            player_tracks.append(frame_dict)

        TrackSanitizer.stitch_fragmented_tracks(
            player_tracks, max_gap_frames=10, max_pixel_distance=120.0
        )

        # Should NOT merge because teams conflict
        for f in range(6, 10):
            assert 2 in player_tracks[f]
            assert 1 not in player_tracks[f]

    def test_stitch_fragmented_tracks_rejects_excessive_distance(self):
        player_tracks = []
        for f in range(10):
            frame_dict = {}
            if 0 <= f <= 4:
                frame_dict[1] = {"bbox": [100.0, 200.0, 140.0, 300.0], "team": 1}
            elif 6 <= f <= 9:
                # Far away: (700, 800) vs (120, 250)
                frame_dict[2] = {"bbox": [700.0, 800.0, 740.0, 900.0], "team": 1}
            player_tracks.append(frame_dict)

        TrackSanitizer.stitch_fragmented_tracks(
            player_tracks, max_gap_frames=10, max_pixel_distance=100.0
        )

        # Should NOT merge because distance is too far
        for f in range(6, 10):
            assert 2 in player_tracks[f]
            assert 1 not in player_tracks[f]
