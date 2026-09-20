# Benchmarking Protocol & Empirical Baselines

This directory stores standardized performance profiling runs, baseline measurements, and environment manifests for `football-cv`.

---

## 1. Frozen Baseline (Pure-Python Pipeline)

* **Recorded:** 2026-09-20
* **Commit:** `2aa1d75`
* **Configuration:** [`configs/benchmark.yaml`](../configs/benchmark.yaml)
* **Dataset:** `input_videos/08fd33_4.mp4` (1920x1080 @ 25 FPS, 100 frames profiled, 5 warmup frames discarded)
* **Model:** `models/best.pt` (Confidence: 0.10, Batch size: 20)

### System Environment
* **Platform:** Windows 10 (Build 26200, AMD64)
* **CPU:** Intel64 Family 6 Model 186 Stepping 2 (10 physical cores, 16 logical threads)
* **RAM:** 15.68 GB (1.81 GB available at run)
* **Accelerator:** CPU only (`CUDA: False`)
* **Python Runtime:** Python 3.11.9, PyTorch 2.8.0+cpu, OpenCV 4.12.0, Ultralytics 8.3.179

### Baseline Results Summary
* **Frames Profiled:** 100 frames
* **Total Duration:** 10.28 seconds
* **Throughput:** **9.73 FPS**

| Stage | Subsystem | Latency (ms/frame) | % of Total | Bottleneck Tier |
|---|---|---|---|---|
| `detection_and_tracking` | YOLOv8 + ByteTrack | 62.63 ms | **60.91%** | Tier 1 (Primary ML Bottleneck) |
| `camera_motion` | Lucas-Kanade Optical Flow | 14.85 ms | **14.44%** | Tier 2 (Primary CV Bottleneck) |
| `team_assignment` | K-Means jersey color clustering | 12.21 ms | **11.87%** | Tier 3 (Analytics Clustering) |
| `rendering` | Overlay annotations & visualizer | 9.34 ms | **9.08%** | Tier 4 (Display/Drawing) |
| `video_decoding` | OpenCV VideoCapture | 3.73 ms | **3.62%** | Tier 5 (I/O & Decoding) |
| `perspective_transform` | 4-point pitch homography | 0.04 ms | **0.04%** | Minimal overhead |
| `possession_assignment` | Spatial player-ball distance | 0.02 ms | **0.02%** | Minimal overhead |
| `ball_interpolation` | Occlusion gap filling | 0.01 ms | **0.01%** | Minimal overhead |
| `speed_distance` | Kinematic displacement | < 0.01 ms | **< 0.01%** | Minimal overhead |
| `analytics_inference` | Event builder & intervals | < 0.01 ms | **< 0.01%** | Minimal overhead |

---

## 2. Reproducing the Baseline

To execute a clean baseline sweep matching this configuration:

```powershell
.venv\Scripts\python.exe -m football_cv.cli benchmark `
  --config configs/benchmark.yaml `
  --num-frames 100 `
  --output-dir benchmarks/
```

Artifacts generated:
* `results.json`: Full JSON report with per-stage latencies, throughput, and run parameters.
* `results.csv`: Tabular CSV row suitable for metric aggregation and plotting.
* `environment.json`: Hardware, CPU, RAM, GPU, OS, and package manifest.
