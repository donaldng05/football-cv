"""
Unit and integration tests for memory-bounded streaming video pipeline.
"""

import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from football_cv.config import load_config
from football_cv.exceptions import VideoProcessingError
from football_cv.pipeline import MatchPipeline
from football_cv.possession.interpolation import StreamingBallInterpolator
from football_cv.rendering.video_writer import IncrementalVideoWriter
from football_cv.utils.video import get_video_properties, stream_video_chunks


class TestStreamVideoChunks:
    def test_stream_video_chunks_slicing(self):
        video_path = "input_videos/08fd33_4.mp4"
        # Request 25 frames in chunks of 10 -> chunks of [10, 10, 5]
        chunks = list(
            stream_video_chunks(video_path, chunk_size=10, start_frame=0, end_frame=25)
        )
        assert len(chunks) == 3
        start_0, chunk_0 = chunks[0]
        start_1, chunk_1 = chunks[1]
        start_2, chunk_2 = chunks[2]

        assert start_0 == 0
        assert len(chunk_0) == 10
        assert start_1 == 10
        assert len(chunk_1) == 10
        assert start_2 == 20
        assert len(chunk_2) == 5

    def test_stream_video_chunks_invalid_chunk_size(self):
        with pytest.raises(VideoProcessingError, match="chunk_size must be positive"):
            list(stream_video_chunks("input_videos/08fd33_4.mp4", chunk_size=0))


class TestIncrementalVideoWriter:
    def test_write_frames_incremental_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "stream_out.avi"
            fps = 25.0
            width, height = 320, 240

            with IncrementalVideoWriter(out_file, fps=fps) as writer:
                # Write 3 chunks of 5 frames
                for c in range(3):
                    frames = [
                        np.full((height, width, 3), fill_value=c * 30, dtype=np.uint8)
                        for _ in range(5)
                    ]
                    writer.write_chunk(frames)

            assert out_file.is_file()
            props = get_video_properties(out_file)
            assert props["width"] == width
            assert props["height"] == height
            assert props["frame_count"] == 15

    def test_write_empty_frame_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test.avi"
            writer = IncrementalVideoWriter(out_file)
            with pytest.raises(VideoProcessingError, match="Cannot write empty"):
                writer.write_frame(np.array([]))

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test.avi"
            writer = IncrementalVideoWriter(out_file)
            writer.write_frame(np.zeros((100, 100, 3), dtype=np.uint8))
            writer.close()
            writer.close()  # second call should be safe


class TestStreamingBallInterpolator:
    def test_sliding_window_lookahead_bridging(self):
        interpolator = StreamingBallInterpolator(limit=5)
        # Chunk 1: 10 frames, ball missing on frame 8, 9
        chunk_1 = [
            {1: {"bbox": [float(i), float(i), float(i + 10), float(i + 10)]}}
            for i in range(8)
        ]
        chunk_1.extend([{}, {}])  # frames 8, 9 missing

        emitted_1 = interpolator.update(chunk_1, is_last_chunk=False)
        # Length of emitted should be len(chunk_1) - limit = 10 - 5 = 5 frames
        assert len(emitted_1) == 5

        # Chunk 2: ball reappears on frame 1 (which is global frame 11)
        chunk_2 = [
            {},  # global frame 10 (still missing)
            {1: {"bbox": [11.0, 11.0, 21.0, 21.0]}},  # global frame 11
            {1: {"bbox": [12.0, 12.0, 22.0, 22.0]}},
        ]
        emitted_2 = interpolator.update(chunk_2, is_last_chunk=True)
        # All remaining buffered frames should now be emitted and bridged
        total_emitted = emitted_1 + emitted_2
        assert len(total_emitted) == 13
        # Check that frames 8, 9, 10 were successfully interpolated across the boundary
        for f in range(8, 11):
            assert 1 in total_emitted[f], (
                f"Frame {f} should be interpolated across chunk boundary"
            )


class TestStreamingPipelineIntegration:
    def test_streaming_pipeline_run_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_video = Path(tmpdir) / "streaming_test.avi"
            cfg = load_config(
                "configs/fast.yaml",
                overrides={
                    "video": {
                        "start_frame": 0,
                        "end_frame": 20,
                        "output_path": str(out_video),
                    },
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 16,
                        "warmup_frames": 10,
                    },
                    "analytics": {"enabled": False},
                },
            )
            pipeline = MatchPipeline(cfg)
            results = pipeline.run()

            assert "tracks" in results
            assert len(results["tracks"]["players"]) == 20
            assert len(results["tracks"]["balls"]) == 20
            assert len(results["camera_movement"]) == 20
            assert out_video.is_file()
            props = get_video_properties(out_video)
            assert props["frame_count"] == 20

    def test_bounded_memory_footprint(self):
        """Verify that streaming execution peak memory is strictly bounded < 500 MB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_video = Path(tmpdir) / "mem_test.avi"
            cfg = load_config(
                "configs/fast.yaml",
                overrides={
                    "video": {
                        "start_frame": 0,
                        "end_frame": 35,
                        "output_path": str(out_video),
                    },
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 16,
                        "warmup_frames": 10,
                    },
                    "analytics": {"enabled": False},
                },
            )
            pipeline = MatchPipeline(cfg)

            tracemalloc.start()
            _ = pipeline.run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / (1024 * 1024)
            # Memory must be comfortably below 500 MB limit
            assert peak_mb < 500.0, (
                f"Peak memory was {peak_mb:.2f} MB, exceeding 500 MB limit"
            )
