"""
Integration test verifying end-to-end analytics event export via MatchPipeline and CLI.
"""

import cv2
import numpy as np

from football_cv.cli import main
from football_cv.config import load_config
from football_cv.pipeline import MatchPipeline


def test_pipeline_export_end_to_end(tmp_path):
    # Create synthetic 6-frame video
    vid_path = tmp_path / "sample_match.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 25.0, (1920, 1080))
    for _ in range(6):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()

    out_video = tmp_path / "out_video.avi"
    export_dir = tmp_path / "structured_analytics"

    overrides = {
        "video": {
            "input_path": str(vid_path),
            "output_path": str(out_video),
            "start_frame": 0,
            "end_frame": 5,
        },
        "tracking": {
            "use_cached_tracks": False,
            "cache_path": str(tmp_path / "stubs.pkl"),
            "camera_movement_cache_path": str(tmp_path / "cam_stubs.pkl"),
        },
        "analytics": {
            "enabled": True,
            "export_events": True,
            "export_dir": str(export_dir),
        },
    }

    config = load_config("configs/fast.yaml", overrides=overrides)
    pipeline = MatchPipeline(config)
    results = pipeline.run()

    # Verify pipeline execution results
    assert "possession_intervals" in results
    assert "events" in results
    assert "export_paths" in results

    export_paths = results["export_paths"]
    expected_files = [
        "player_tracking.csv",
        "player_tracking.json",
        "ball_tracking.csv",
        "ball_tracking.json",
        "possession_intervals.csv",
        "possession_intervals.json",
        "events.csv",
        "events.json",
        "metadata.json",
    ]

    for filename in expected_files:
        target_file = export_dir / filename
        assert target_file.is_file(), f"Expected file missing: {target_file}"
        assert target_file.stat().st_size > 0

    # Test report CLI validation on the exported player tracking JSON
    report_out = tmp_path / "report_out"
    ret = main(
        [
            "report",
            "--tracks",
            str(export_paths["player_tracking_json"]),
            "--output-dir",
            str(report_out),
        ]
    )
    assert ret == 0
