# Project Attribution & Intellectual Boundaries

## 1. Upstream Foundation & Tutorial Origin

`football-cv` originates from the comprehensive computer vision tutorial and reference implementation published by **Abdullah Tarek** and **Roboflow**:
- **Reference Tutorial:** *Football AI Analysis with Computer Vision and Deep Learning*
- **Tutorial Author:** Abdullah Tarek
- **Organization:** Roboflow

The baseline architecture established a functional proof-of-concept demonstrating object detection with YOLOv8, multi-object tracking with ByteTrack, jersey color clustering with K-Means, optical-flow-based camera movement compensation, and basic perspective transformation.

---

## 2. Distinction Between Adapted Code and Original Work

To maintain engineering transparency and professional academic/industry credibility, this project clearly delineates between the adapted baseline code and the original engineering contributions introduced in this repository.

### A. Adapted Upstream Components
The foundational computer vision pipeline adapts concepts and initial logic from the tutorial:
1. **Object Detection & ByteTrack Hook:** YOLOv8 bounding box predictions and conversion into `supervision` ByteTrack data structures (`trackers/tracker.py`).
2. **Jersey Color Clustering:** K-Means clustering ($k=2$) applied to cropped player jersey patches to classify teams (`team_assigner/team_assigner.py`).
3. **Camera Motion Estimation:** Lucas-Kanade sparse optical flow tracking features across the frame perimeter to compensate for pan/tilt (`camera_movement_estimator/camera_movement_estimator.py`).
4. **Perspective Transformation:** 4-point homography mapping broadcast pixel coordinates to a real-world metric pitch subsection (`view_transformer/view_transformer.py`).
5. **Basic Proximity Ball Control:** Proximity assignment between detected ball and player foot/bounding-box positions (`player_ball_assigner/player_ball_assigner.py`).

### B. Original Additions & Advanced Engineering
The following systems, architectures, and algorithms were designed and implemented independently as extensions to transform the tutorial into a production-grade analytics package:
1. **Production Package Architecture (`src/football_cv`):**
   - Refactored monolithic scripts into a modular, installable Python package adhering to modern packaging standards (`pyproject.toml`).
   - Decoupled analytics computation and geometric algorithms from OpenCV frame drawing loops.
   - Replaced memory-heavy frame arrays with streamed, generator-based video I/O.
2. **Configuration & Unified CLI (`football-cv`):**
   - Hierarchical YAML configuration with schema validation (`configs/default.yaml`, `configs/fast.yaml`, `configs/high_accuracy.yaml`).
   - CLI subcommands: `analyze`, `report`, `benchmark`, and `validate` with parameter overrides.
3. **Structured Event Data Export:**
   - Standalone export of player tracks, ball trajectories, possession intervals, and candidate transition events to CSV and JSON formats, enabling downstream analytics without re-rendering video.
4. **Original Football Analytics Layer:**
   - **Possession Duration Heatmaps:** 2D spatial density mapping onto standardized 105m × 68m pitch dimensions, duration-weighted by frame interval ($\Delta t$).
   - **Possession State Machine with Temporal Hysteresis:** Filtering close-proximity flicker with an $N$-frame stability buffer.
   - **Pass Network Inference:** Transition event classification (`candidate_pass`, `turnover`, `recovery`, `loose_ball`, `uncertain_transition`) and directed network graph generation with weighted edges.
5. **Testing, CI/CD & Verification:**
   - Complete `pytest` unit test suite covering geometry, configuration, coordinate transformations, and possession logic.
   - Lightweight synthetic CPU smoke test running in < 15 seconds on GitHub Actions CI runners.
   - Automated code formatting and linting via `ruff`.
6. **Empirical Benchmarking & Error Taxonomy:**
   - Profiling pipeline stages to measure per-stage latency (ms) and overall throughput (FPS).
   - Documented failure taxonomy (ball detection dropout, track switches, optical flow drift) and implemented mitigations.

---

## 3. Licensing Summary

| Component | License / Provenance |
| :--- | :--- |
| **`football-cv` Source Code** | MIT License (Copyright © 2024–2026 Quy Duong) |
| **Tutorial Base Code** | Adapted from Roboflow / Abdullah Tarek open tutorial |
| **YOLOv8 Detection Weights (`models/best.pt`)** | Trained on Roboflow football tracking dataset (Non-commercial / Educational evaluation) |
| **Sample Footage (`input_videos/08fd33_4.mp4`)** | Short broadcast clip used solely for non-commercial demonstration and research benchmarks |
