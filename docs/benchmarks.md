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

---

## 6. Empirical Baseline Profile & Bottleneck Analysis

* **Recorded:** 2026-09-20
* **Commit:** `2aa1d75`
* **Configuration:** [`configs/benchmark.yaml`](../configs/benchmark.yaml)
* **Dataset:** `input_videos/08fd33_4.mp4` (1920x1080 @ 25 FPS, 100 frames profiled, 5 warmup frames discarded)
* **Hardware:** Intel64 Family 6 Model 186 Stepping 2 (16 logical threads), 15.68 GB RAM, Windows 10
* **Accelerator:** CPU only (`CUDA: False`), PyTorch 2.8.0+cpu, OpenCV 4.12.0
* **Raw Artifacts:** [`benchmarks/baseline_results.json`](../benchmarks/baseline_results.json)

### Stage Latency Breakdown

| Stage | Subsystem | Latency (ms/frame) | % of Total Time | Engineering Bottleneck Tier |
|---|---|---|---|---|
| `detection_and_tracking` | YOLOv8 + ByteTrack | 62.63 ms | **60.91%** | Tier 1 (Primary ML Bottleneck) |
| `camera_motion` | Lucas-Kanade Optical Flow | 14.85 ms | **14.44%** | Tier 2 (Primary Vision Bottleneck) |
| `team_assignment` | K-Means jersey color clustering | 12.21 ms | **11.87%** | Tier 3 (Analytics Clustering) |
| `rendering` | Overlay annotations & visualizer | 9.34 ms | **9.08%** | Tier 4 (Display/Drawing) |
| `video_decoding` | OpenCV VideoCapture | 3.73 ms | **3.62%** | Tier 5 (I/O & Decoding) |
| `perspective_transform` | 4-point pitch homography | 0.04 ms | **0.04%** | Low (< 0.1 ms) |
| `possession_assignment` | Spatial player-ball distance | 0.02 ms | **0.02%** | Low (< 0.1 ms) |
| `ball_interpolation` | Occlusion gap filling | 0.01 ms | **0.01%** | Low (< 0.1 ms) |
| `speed_distance` | Kinematic displacement | < 0.01 ms | **< 0.01%** | Low (< 0.1 ms) |
| `analytics_inference` | Event builder & intervals | < 0.01 ms | **< 0.01%** | Low (< 0.1 ms) |
| **Total Pipeline** | **End-to-End Execution** | **102.83 ms** | **100.0%** | **Throughput: 9.73 FPS** |

### Where is the time actually going?

1. **Model Inference & Tracking (60.91% of latency):**
   YOLOv8 forward passes and ByteTrack bounding-box associations account for over 60% of total frame latency. This empirical finding confirms that purely rewriting post-processing in C++ will not yield dramatic end-to-end speedups on its own; tackling this tier requires model deployment via **ONNX Runtime C++** (Phase 6).
2. **Optical Flow Camera Motion Estimation (14.44% of latency):**
   Lucas-Kanade optical flow on perimeter pixels is the largest pure computer-vision component (14.85 ms/frame). Migrating this calculation to native C++ with OpenCV (Phase 2 Component C) targets the highest-latency vision algorithm in the pipeline.
3. **Jersey Color Clustering (11.87% of latency):**
   Scikit-learn K-Means clustering on cropped player jersey patches adds non-trivial per-frame overhead when tracking multiple players.
4. **Drawing & Visual Overlays (9.08% of latency):**
   Rendering bounding boxes, player speed badges, and possession indicators directly onto 1080p frames takes ~9.3 ms/frame.
5. **Geometry, Homography & Analytics (< 0.1% of latency):**
   Mathematical operations (Euclidean distance, bounding box centers, 4-point homography projection) execute in microseconds per frame. While migrating geometry to C++ (Phase 2 Component A & B) will not drastically shift end-to-end FPS, it provides the essential, testable foundation for typed C++ data structures (`Point2D`, `BoundingBox`, `PerspectiveTransformer`) and validates pybind11 interoperability without architectural complexity.

---

## 7. Numerical Parity Validation (Phase 4)

To guarantee that moving vision algorithms into C++ does not introduce subtle mathematical drift or tracking errors, the pipeline undergoes strict numerical parity testing against the pure-Python reference implementation.

### Tolerance Specifications

| Domain | Operation | Mathematical Tolerance | Notes |
| :--- | :--- | :--- | :--- |
| **Geometry** | BBox Dimensions & Areas | `atol = 1e-9` | Exact floating-point parity |
| **Geometry** | Centers & Foot Positions | `atol = 1e-9` (float), exact `int()` | C++ preserves continuous sub-pixel precision; Python legacy truncated to `int()` |
| **Geometry** | Euclidean & XY Distance | `atol = 1e-9` | IEEE 754 float64 parity |
| **Homography** | 3x3 Transformation Matrix | `rtol = 1e-5, atol = 1e-5` | Analytic 8x8 linear solve vs OpenCV `cv::getPerspectiveTransform` |
| **Projection** | Pitch Metric Coordinates | `atol = 1e-4` meters (0.1 mm) | Verified on 500-point grid inside calibrated pitch polygon |
| **Containment** | Polygon Boundary Test | Exact boolean parity | C++ boundary segment cross-product matches `cv2.pointPolygonTest >= 0` |
| **Optical Flow**| Synthetic Feature Shifts | `atol = 1e-9` | Exact displacement, sub-threshold filter, and scene cut detection |
| **Optical Flow**| Real Match Video (50 frames)| `atol = 1e-5` | Verified frame-by-frame on `input_videos/08fd33_4.mp4` |
| **Tracks Mutation**| Transformed / Adjusted Tracks | `atol = 1e-4` | Full multi-frame pipeline track dictionary parity |

### Key Engineering Findings
1. **Sub-Pixel Accuracy Retention**: Legacy Python implementations of `get_center_of_bbox` and `get_foot_position` truncated coordinates using `int()`, introducing up to 0.5 px quantization error. The C++ `football_cv_core` preserves continuous double-precision coordinates while maintaining exact integer compatibility.
2. **Boundary Classification Parity**: Standard ray-casting algorithms can miss points lying exactly on vertices or collinear boundary segments. Adding segment cross-product and dot-product tests in C++ aligns boundary inclusion exactly with OpenCV's `pointPolygonTest >= 0`.
3. **Optical Flow Stability**: Frame-by-frame camera motion translation vectors on 50 consecutive 1080p match video frames demonstrate zero numerical divergence between Python and native C++ backend estimators.
