"""
End-to-end integration smoke test for MatchPipeline.
"""

from football_cv.config import load_config
from football_cv.pipeline import MatchPipeline
from football_cv.utils.video import read_video


def test_pipeline_smoke_end_to_end(tmp_path):
    out_video = tmp_path / "smoke_out.avi"
    overrides = {
        "video": {
            "start_frame": 0,
            "end_frame": 3,
            "output_path": str(out_video),
        },
        "tracking": {
            "use_cached_tracks": False,
            "cache_path": str(tmp_path / "track_stubs.pkl"),
            "camera_movement_cache_path": str(tmp_path / "cam_stubs.pkl"),
        },
    }
    config = load_config("configs/fast.yaml", overrides=overrides)
    pipeline = MatchPipeline(config)

    frames = read_video(config.video.input_path, start_frame=0, end_frame=3)
    assert len(frames) == 3

    results = pipeline.process(frames)
    assert "tracks" in results
    assert len(results["tracks"]["players"]) == 3
    assert len(results["camera_movement"]) == 3
    assert len(results["team_ball_control"]) == 3

    # Test rendering
    rendered = pipeline.render(
        frames=frames,
        tracks=results["tracks"],
        camera_movement=results["camera_movement"],
        team_ball_control=results["team_ball_control"],
        output_path=str(out_video),
    )
    assert len(rendered) == 3
    assert out_video.is_file()
    assert out_video.stat().st_size > 0
