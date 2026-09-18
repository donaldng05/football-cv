# Rigorous Error Analysis & Failure Taxonomy

Comprehensive failure taxonomy, reproducible benchmark cases, and quantitative mitigation evaluations across detection, tracking, perspective/motion, and tactical analytics stages in `football-cv`.

---

## 1. Executive Summary

Computer vision models deployed on broadcast sports footage operate under adverse visual conditions: rapid player decelerations, ball motion blur, multi-player occlusions during scrums, camera whip pans, and unannounced broadcast cuts. Without systematic categorization and defensive algorithmic filters, small upstream perceptual errors cascade into catastrophic tactical misattributions (e.g. false passes or inverted possession metrics).

Phase 10 of `football-cv` establishes:
1. **Four-tier Failure Taxonomy**: Spanning Detection, Tracking, Perspective & Motion, and Analytics.
2. **Standard Diagnostic Benchmark Suite**: Five reproducible synthetic and empirical edge cases.
3. **Validated Algorithmic Mitigations**:
   - Kinematic displacement validation rejecting near-zero transfer tracker identity swaps.
   - Superhuman pass velocity threshold filtering (> 45.0 m/s / 162 km/h).
   - Optical flow discontinuity detection and vector reset across broadcast scene cuts (> 25 px/frame).
   - Possession temporal hysteresis buffer (≥ 2 frames) suppressing scrum assignment flicker.
   - Temporal gap-filling buffer (≤ 2 frames) bridging short ball detection dropouts.
4. **Automated Diagnostic CLI Command**: `football-cv error-analysis` generating machine-readable JSON and CSV reports.

---

## 2. Failure Taxonomy

```text
Pipeline Failures
├── 1. Detection
│   ├── Small-object ball miss / dropout (motion blur, low pixel footprint)
│   ├── False-positive ball detection (sock tape, referee footwear, white pitch markings)
│   └── Referee misclassified as outfield player
├── 2. Tracking
│   ├── Track-ID switch (identity swap on same physical player)
│   ├── Re-identification loss through dense multi-player occlusion
│   └── Superhuman teleportation jump (erroneous track association across distance)
├── 3. Perspective & Motion
│   ├── Optical-flow drift from player movement within perimeter margin
│   ├── Broadcast scene-cut breakdown (optical flow accumulation spike)
│   └── Non-planar pitch edge perspective distortion
└── 4. Tactical Analytics & Event Inference
    ├── Close-proximity possession flicker (1-frame turnover oscillation)
    └── Track swap registered as candidate pass between teammates
```

---

## 3. Curated Benchmark Failure Cases & Mitigations

| Case ID | Stage | Failure Mode | Root Cause | Baseline Metric | Mitigated Metric | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`TS_PASS_01`** | Tracking | Track-Swap False Pass | Track ID switch on same player (Δd < 1.5m, Δt ≤ 2 frames) | 1.0 false passes | 0.0 false passes | **100.0%** |
| **`SH_VEL_02`** | Tracking | Superhuman Velocity | Bounding box teleportation across pitch (v > 45 m/s) | 1.0 false passes | 0.0 false passes | **100.0%** |
| **`POSS_FLK_03`** | Analytics | Possession Flicker | Equidistant scrum contest causing 1-frame oscillation | 2.0 false turnovers | 0.0 false turnovers | **100.0%** |
| **`CAM_CUT_04`** | Perspective | Camera Cut Drift | Broadcast scene cut causing optical flow accumulation jump | 39.55 px drift | 0.00 px drift | **100.0%** |
| **`BALL_DROP_05`** | Detection | Ball Detection Dropout | Motion blur or partial occlusion dropping ball for 1–2 frames | 2.0 intervals | 1.0 interval | **50.0%** |

---

## 4. Algorithmic Mitigation Implementations

### 4.1. Kinematic Pass Validation & Track-Swap Rejection

* **File**: [`src/football_cv/analytics/event_builder.py`](../src/football_cv/analytics/event_builder.py)
* **Problem**: When a tracker loses a player for a single frame and re-assigns a new track ID to the same physical person, `EventBuilder` previously saw an intra-team transition between two distinct IDs and logged a candidate pass.
* **Mitigation**:
  $$\Delta d = \sqrt{(x_{\text{end}} - x_{\text{start}})^2 + (y_{\text{end}} - y_{\text{start}})^2}, \quad v = \frac{\Delta d}{\Delta t}$$
  - If $\Delta t \le 2 / \text{fps}$ and $\Delta d < 1.5\text{ m}$, the transition is classified as `EventType.EXCLUDED` (track swap artifact) with confidence $0.95$.
  - If $v > 45.0\text{ m/s}$ ($162\text{ km/h}$), the transition is classified as `EventType.UNCERTAIN_TRANSITION` with confidence $0.40$.

### 4.2. Possession Hysteresis & Temporal Smoothing

* **File**: [`src/football_cv/possession/events.py`](../src/football_cv/possession/events.py)
* **Problem**: In dense goalmouth scrums, player-ball Euclidean distances fluctuate by millimeters each frame, causing single-frame turnover ping-pong. Additionally, 1-frame ball dropouts artificially split continuous possessions into fragmented intervals.
* **Mitigation**:
  - `hysteresis_frames >= 2`: Reassigns transient opponent control runs lasting $< 2$ frames that revert immediately to the original player.
  - `max_gap_frames >= 2`: Bridges missing ball frames ($\le 2$ frames) between the same possessor into a single contiguous interval.

### 4.3. Camera Cut Discontinuity Detection

* **File**: [`src/football_cv/camera_motion/estimator.py`](../src/football_cv/camera_motion/estimator.py)
* **Problem**: Broadcast scene cuts (e.g. switching to a goal replay or bench camera) cause perimeter Lucas-Kanade optical flow to track across unrelated scenes, introducing massive translation drift into coordinate stabilization.
* **Mitigation**:
  - `scene_cut_threshold`: When maximum tracked feature displacement exceeds the cut threshold ($25.0\text{ px/frame}$), camera translation for that frame is clamped to $(0.0, 0.0)$, and corner features are refreshed from the new frame.

---

## 5. Diagnostic CLI Execution

The diagnostic suite can be executed on any environment to verify pipeline resilience and export reproducible reports:

```bash
# Run full diagnostic suite with default YAML configuration
football-cv error-analysis --config configs/default.yaml --output-dir reports/error_analysis
```

### Generated Artifacts
1. **`reports/error_analysis/error_report.json`**: Complete execution metadata, before/after parameters, and case statuses.
2. **`reports/error_analysis/mitigation_summary.csv`**: Tabular metric deltas and percentage improvements suitable for CI assertions and dashboards.

---

## 6. Reproducing Failure Scenarios in Python

```python
from football_cv.analytics import FailureCaseEvaluator, ErrorAnalysisExporter

evaluator = FailureCaseEvaluator(fps=25.0)
report = evaluator.run_suite()

# Print formatted summary table
print(ErrorAnalysisExporter.format_terminal_table(report))

# Export reports
ErrorAnalysisExporter.export_json(report, "reports/error_analysis/error_report.json")
ErrorAnalysisExporter.export_csv(
    report, "reports/error_analysis/mitigation_summary.csv"
)
```
