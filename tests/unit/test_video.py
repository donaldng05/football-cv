"""
Unit tests for video reading and streaming utilities.
"""

import numpy as np
import pytest

from football_cv.exceptions import VideoProcessingError
from football_cv.utils.video import (
    get_video_properties,
    read_video,
    save_video,
    stream_video_frames,
)


class TestVideoUtilities:
    def test_get_video_properties(self):
        props = get_video_properties("input_videos/08fd33_4.mp4")
        assert props["width"] == 1920
        assert props["height"] == 1080
        assert props["fps"] == 25.0
        assert props["frame_count"] == 750

    def test_stream_video_frames_sliced(self):
        stream = stream_video_frames(
            "input_videos/08fd33_4.mp4", start_frame=0, end_frame=3
        )
        frames = list(stream)
        assert len(frames) == 3
        idx, frame = frames[0]
        assert idx == 0
        assert frame.shape == (1080, 1920, 3)

    def test_read_video_bounded(self):
        frames = read_video("input_videos/08fd33_4.mp4", start_frame=10, end_frame=14)
        assert len(frames) == 4
        assert frames[0].shape == (1080, 1920, 3)

    def test_save_and_reopen_video(self, tmp_path):
        dummy_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        out_vid = tmp_path / "test.avi"
        save_video(dummy_frames, out_vid, fps=10.0)
        assert out_vid.is_file()
        assert out_vid.stat().st_size > 0

    def test_save_empty_frames_raises(self, tmp_path):
        with pytest.raises(VideoProcessingError, match="No frames provided"):
            save_video([], tmp_path / "empty.avi")
