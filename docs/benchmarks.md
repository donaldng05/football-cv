# Benchmarking Framework & Performance Profiling

## 1. Overview & Objectives

The `football-cv` benchmarking framework provides systematic, reproducible measurement of pipeline latency, throughput (FPS), and stage-by-stage computational trade-offs across hardware targets and configuration settings.

Key goals:
* **Stage-by-Stage Profiling:** Quantify where execution time is spent across detection, tracking, optical flow, homography, and rendering.
* **Reproducibility:** Capture complete hardware and software environment manifests (`reports/benchmarks/environment.json`).
* **Automated Parameter Sweeps:** Sweep detection confidence thresholds ($0.10, 0.20, 0.30, 0.40, 0.50$) and batch sizes.
* **Warmup Isolation:** Discard initial cold-start frames (e.g. CUDA kernel JIT compilation, OpenCV video buffer initialization) to record steady-state metrics.

---

## 2. Profiling Stages

The pipeline instruments 10 distinct phases:

| Stage Name | Description | Key Subsystems |
| :--- | :--- | :--- |
| `video_decoding` | Sequential frame reading and BGR array conversion | OpenCV `VideoCapture` |
| `detection_and_tracking` | YOLOv8 batch inference and ByteTrack identity association | Ultralytics, Supervision |
| `camera_motion` | Frame-margin Lucas-Kanade optical flow estimation | OpenCV `calcOpticalFlowPyrLK` |
| `perspective_transform` | 4-point homography projection to 2D pitch coordinates | OpenCV `warpPerspective` |
| `ball_interpolation` | Gap filling for occluded ball detections | Pandas linear interpolation |
| `speed_distance` | Rolling-window displacement and kinematics | NumPy metric transformations |
| `team_assignment` | Jersey patch extraction and 2-stage K-Means clustering | Scikit-learn K-Means |
| `possession_assignment`| Euclidean distance matching between ball and player feet | Proximity spatial assigner |
| `analytics_inference` | Hysteresis temporal filtering and candidate event classification | State machine event builder |
| `rendering` | Overlaying player ellipses, trajectories, and speed badges | OpenCV drawing primitives |

---

## 3. CLI Usage

### Basic Benchmark
Run a 50-frame benchmark using the fast configuration:
```bash
football-cv benchmark --config configs/fast.yaml --num-frames 50
```

### Confidence Threshold Sweep
Sweep across detection confidence thresholds:
```bash
football-cv benchmark --config configs/fast.yaml --num-frames 50 --confidences 0.10,0.20,0.30,0.40,0.50
```

### Full Hardware Benchmark with Warmup
Evaluate live inference without track stubs on custom devices:
```bash
football-cv benchmark --config configs/benchmark.yaml --num-frames 100 --no-use-stubs --warmup 10 --output-dir reports/benchmarks
```

---

## 4. Output Artifacts

Each benchmark execution outputs three standardized artifacts to `--output-dir`:

1. **`results.csv`**: Tabular performance metrics for cross-run regression tracking:
   ```csv
   benchmark_id,run_name,total_duration_seconds,throughput_fps,detection_and_tracking_ms,camera_motion_ms,...
   ```
2. **`results.json`**: Hierarchical summary with complete per-stage duration and percentage breakdowns.
3. **`environment.json`**: Hardware specifications (CPU cores, RAM, GPU model, CUDA version) and library version stamps.

---

## 5. CI Automation

Benchmarking is automated via GitHub Actions in `.github/workflows/benchmark.yml`, allowing on-demand profiling runs on hosted Linux runners or upon release tags.
