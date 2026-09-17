# Full Next-Steps Plan for `football-cv`

This plan will transform the repository from a lightly customized tutorial implementation into a credible, reproducible, portfolio-grade computer vision project. The work is deliberately scoped around engineering quality, original football analytics, and rigorous evaluation, without adding unnecessary cloud infrastructure or unrelated features.

---

## 1. Final Project Objective

By the end of the project, the repository should demonstrate four things clearly:

1. You understand the original computer vision pipeline.
2. You can refactor an existing codebase into maintainable software.
3. You can design and implement original football analytics.
4. You can evaluate an ML system honestly, including its failures.

The final project should allow another developer to:

```bash
git clone <repository-url>
cd football-cv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
football-cv analyze \
  --input examples/sample_match.mp4 \
  --output outputs/annotated_match.mp4

```

They should also be able to generate analytics:

```bash
football-cv report \
  --tracks outputs/tracks.json \
  --output-dir outputs/report

```

And run quality checks:

```bash
pytest
ruff check .
ruff format --check .

```

No source-code edits should be required to change the input video, model, output location, confidence threshold, frame rate, or analytics settings.

---

## 2. Scope Boundaries

### Included

* Production-quality Python package structure
* YAML configuration
* Command-line interface
* Input validation
* Structured logging
* Reproducible dependency management
* Unit and integration tests
* Continuous integration
* Model and video asset documentation
* Base-tutorial attribution
* Architecture documentation
* Annotated demonstration video
* Structured event-data export
* Possession heatmaps
* Pass networks
* Model and configuration benchmarks
* Error-analysis report
* Limitations and methodology documentation

### Explicitly Excluded

Do not add the following unless everything else is complete:

* Web dashboard
* Mobile application
* Cloud deployment
* Database
* User authentication
* REST API
* Kubernetes
* Real-time streaming
* Microservices
* Multiple frontend frameworks
* Complex experiment-tracking infrastructure
* A second unrelated ML model

These would increase scope without proportionally increasing the hiring signal.

---

## 3. Target Repository Structure

```text
football-cv/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml
│       └── benchmark.yml
│
├── configs/
│   ├── default.yaml
│   ├── fast.yaml
│   └── high_accuracy.yaml
│
├── docs/
│   ├── architecture.md
│   ├── attribution.md
│   ├── benchmark.md
│   ├── error_analysis.md
│   ├── methodology.md
│   └── assets/
│       ├── architecture-diagram.png
│       ├── demo.gif
│       ├── possession-heatmap.png
│       ├── pass-network.png
│       └── failure-cases/
│
├── examples/
│   ├── README.md
│   └── sample_config.yaml
│
├── models/
│   └── README.md
│
├── reports/
│   ├── README.md
│   ├── benchmarks/
│   ├── figures/
│   └── error_analysis/
│
├── scripts/
│   ├── download_assets.py
│   ├── create_demo.py
│   ├── run_benchmark.py
│   └── validate_installation.py
│
├── src/
│   └── football_cv/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── exceptions.py
│       ├── logging_config.py
│       ├── pipeline.py
│       │
│       ├── tracking/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   ├── tracker.py
│       │   └── schemas.py
│       │
│       ├── camera_motion/
│       │   ├── __init__.py
│       │   └── estimator.py
│       │
│       ├── perspective/
│       │   ├── __init__.py
│       │   └── transformer.py
│       │
│       ├── possession/
│       │   ├── __init__.py
│       │   ├── assigner.py
│       │   ├── interpolation.py
│       │   └── events.py
│       │
│       ├── teams/
│       │   ├── __init__.py
│       │   └── classifier.py
│       │
│       ├── movement/
│       │   ├── __init__.py
│       │   └── speed_distance.py
│       │
│       ├── analytics/
│       │   ├── __init__.py
│       │   ├── heatmap.py
│       │   ├── pass_network.py
│       │   ├── event_builder.py
│       │   └── pitch.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── benchmark.py
│       │   ├── metrics.py
│       │   └── failure_analysis.py
│       │
│       ├── rendering/
│       │   ├── __init__.py
│       │   ├── annotations.py
│       │   ├── video_writer.py
│       │   └── report_figures.py
│       │
│       └── utils/
│           ├── geometry.py
│           ├── validation.py
│           └── video.py
│
├── tests/
│   ├── fixtures/
│   ├── integration/
│   │   ├── test_cli.py
│   │   └── test_pipeline_smoke.py
│   └── unit/
│       ├── test_config.py
│       ├── test_geometry.py
│       ├── test_possession_assigner.py
│       ├── test_possession_events.py
│       ├── test_speed_distance.py
│       ├── test_pass_network.py
│       └── test_view_transformer.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
└── pyproject.toml

```

*Move code incrementally into this target structure while keeping the pipeline operational.*

---

## 4. Implementation Principles

### Preserve Behavior During Refactoring

Do not simultaneously reorganize code, change algorithms, add visualizations, and tune performance in one commit.

1. Capture current behavior.
2. Add smoke tests where possible.
3. Move code without changing behavior.
4. Verify that output remains equivalent.
5. Add new behavior afterward.

### Prefer Small Modules Over Unnecessary Abstractions

Use abstractions only where they provide a clear benefit:

* Swapping model variants
* Testing deterministic logic
* Separating I/O from transformations
* Comparing multiple configurations
* Exporting structured results

### Keep Analytical Claims Proportional to Evidence

Do not call an inferred possession transition a confirmed pass unless the inference method supports that claim. Use disciplined terminology:

* *“Detected player”*
* *“Estimated pitch position”*
* *“Assigned possession”*
* *“Candidate pass”*
* *“Likely turnover”*
* *“Uncertain transition”*

### Never Hide the Tutorial Origin

Attribution establishes engineering credibility:

> *"I began with a tutorial-based football tracking pipeline, refactored it into a tested and configurable package, then designed an analytics and evaluation layer that was not present in the original implementation."*

---

## 5. Phase 0: Establish a Baseline

* **Goal:** Understand and preserve current behavior before making architectural changes.
* **Estimated Effort:** 0.5 days

### Tasks

* **5.1 Create baseline branch and tag:**
```bash
git checkout -b refactor/production-foundation
git tag pre-production-refactor

```


* **5.2 Run the existing pipeline & record environment:** Python version, OS, CPU, GPU, CUDA version, input/output paths, resolution, frame count, runtime, and FPS.
* **5.3 Capture baseline artifacts:** Output video clip, 5 representative annotated frames, console logs, configuration values, model checksum, and execution commands.
* **5.4 Inventory dependencies:** Separate runtime dependencies (`ultralytics`, `opencv-python`, `supervision`, `scikit-learn`, `numpy`, `pandas`, `pyyaml`) from dev tooling.
* **5.5 Document pipeline stages:**
1. Load video
2. Load object detector
3. Detect and track objects
4. Estimate camera movement
5. Correct track positions
6. Transform positions into pitch coordinates
7. Interpolate missing ball detections
8. Calculate speed and distance
9. Assign teams
10. Assign possession
11. Render annotations
12. Write output video



### Acceptance Criteria

* Current pipeline executes successfully end-to-end.
* Baseline output clip and configuration state are archived.
* Hardcoded runtime values are cataloged.

**Suggested Commit:** `docs: capture baseline pipeline behavior and environment`

---

## 6. Phase 1: Transparency and Repository Hygiene

* **Goal:** Resolve baseline credibility issues prior to code refactoring.
* **Estimated Effort:** 2–3 hours

### Tasks

* **6.1 Add repository metadata:** Add topics (`computer-vision`, `football-analytics`, `sports-analytics`, `yolov8`, `object-tracking`, `opencv`, `bytetrack`, `python`, `machine-learning`) and a crisp description.
* **6.2 Add licensing:** Explicitly verify licensing boundaries for source code, tutorial base, model weights, and sample match footage.
* **6.3 Document attribution (`docs/attribution.md`):** Clearly distinguish adapted upstream components from independent additions (CLI, CI, heatmaps, pass network, benchmarks).
* **6.4 Document models (`models/README.md`):** Record architecture, checkpoint origins, classes, SHA-256 checksums, and licensing.
* **6.5 Document sample video:** Provenance, redistribution limits, resolution, duration, and fallback instructions for user-provided footage.
* **6.6 Clean installation instructions:** Align README instructions with package dependencies.

### Acceptance Criteria

* Repository metadata, licensing, and attribution documents are published.
* Model and asset provenance are fully traceable.

**Suggested Commits:**

```bash
docs: credit original pipeline source and define project scope
docs: document model and sample asset provenance
chore: add repository license and contribution metadata
docs: correct installation instructions

```

---

## 7. Phase 2: Configuration and Command-Line Interface

* **Goal:** Eliminate hardcoded execution paths and parameters.
* **Estimated Effort:** 1 day

### Tasks

* **7.1 Design configuration schema (`configs/default.yaml`):**

```yaml
model:
  path: models/best.pt
  confidence: 0.10
  device: auto
  batch_size: 20

video:
  input_path: input_videos/sample.mp4
  output_path: outputs/annotated.mp4
  frame_rate: null
  start_frame: 0
  end_frame: null

tracking:
  use_cached_tracks: false
  cache_path: outputs/cache/tracks.pkl

possession:
  max_player_ball_distance: 70
  minimum_control_frames: 3
  maximum_missing_ball_frames: 10

movement:
  speed_window_frames: 5
  minimum_displacement: 0.0

analytics:
  enabled: true
  export_events: true
  heatmap:
    enabled: true
    grid_width: 60
    grid_height: 40
  pass_network:
    enabled: true
    maximum_transition_frames: 15

logging:
  level: INFO
  log_file: outputs/logs/football-cv.log

```

* **7.2 Implement typed configuration loader:** Validate ranges (e.g., $0 \le \text{confidence} \le 1$), resolve relative paths, check file existence, and generate explicit error messages.
* **7.3 Build CLI (`football-cv`):**
* Commands: `analyze`, `report`, `benchmark`, `validate`
* Support CLI overrides with precedence: `CLI overrides > selected YAML file > defaults`.


* **7.4 Provide validation command:** `football-cv validate --config configs/default.yaml` checks file paths, GPU/CPU availability, and video decodability without running full inference.

### Acceptance Criteria

* No user-specific hardcoded paths remain in application logic.
* Faulty configurations emit descriptive errors instead of tracebacks.

**Suggested Commits:**

```bash
refactor: add validated YAML configuration loader
feat: add command-line interface for video analysis
feat: support CLI overrides for pipeline configuration
feat: validate model video and output paths
test: cover configuration parsing and validation

```

---

## 8. Phase 3: Package and Architecture Refactor

* **Goal:** Transform repository into an installable, modular Python package.
* **Estimated Effort:** 2–3 days

### Tasks

* **8.1 Package structure:** Define `pyproject.toml` supporting `pip install -e ".[dev]"`.
* **8.2 Migrate modules into `src/football_cv`:** Transition utilities, geometry, possession, tracking, and rendering sequentially.
* **8.3 Decouple processing from rendering:** Ensure algorithms emit structured records before feeding visualization engines.
* **8.4 Define canonical internal schemas:**

```text
frame_index | timestamp_seconds | object_type | track_id | team_id | bounding_box | image_position | adjusted_position | pitch_position | confidence | speed | distance | has_ball

```

* **8.5 Replace silent exceptions:** Remove bare `except: pass` blocks; add domain-specific exceptions.
* **8.6 Implement structured logging:** Add configurable logging per stage (DEBUG, INFO, WARNING, ERROR).

### Acceptance Criteria

* CLI runs from an installed wheel or editable package.
* Computation modules operate independently of OpenCV rendering loops.

**Suggested Commits:**

```bash
build: convert repository to installable Python package
refactor: move geometry and video utilities into package
refactor: extract pipeline orchestration from entry point
refactor: separate analytics computation from rendering
refactor: define structured tracking and possession records
feat: add structured application logging

```

---

## 9. Phase 4: Testing Foundation

* **Goal:** Validate deterministic components with automated unit and integration tests.
* **Estimated Effort:** 2 days

### Target Modules & Test Suites

* **Configuration:** Validation logic, path handling, boundary checks, and override precedence.
* **Geometry:** Center coordinates, Euclidean distance calculations, and bounding-box validation.
* **Possession Assigner:** Threshold cutoffs, multi-player ties, empty detections, and identity switches.
* **Perspective Transform:** Known synthetic coordinate transforms (pitch bounds, center point, singular matrix handling).
* **Speed & Distance:** Constant-speed movement, track dropouts, displacement filtering, and noise spikes.
* **Ball Interpolation:** Gap limitations, boundary edge cases, and excessive frame-drop rejections.
* **Analytics Events:** Candidate pass transitions, turnover flags, and recovery conditions.
* **Integration Smoke Test:** End-to-end processing across a synthetic 5-to-10 frame sequence without GPU requirements.

### Acceptance Criteria

* All tests pass offline in CPU environments.
* Smoke tests execute in under 30 seconds on CI runners.

**Suggested Commits:**

```bash
test: cover geometry utility edge cases
test: cover possession assignment behavior
test: validate perspective transformation
test: cover speed and distance calculations
test: cover missing ball interpolation
test: add lightweight pipeline smoke test

```

---

## 10. Phase 5: Code Quality Automation and CI

* **Goal:** Automate checks on all code changes.
* **Estimated Effort:** 0.5–1 day

### Tasks

* **10.1 Configure code quality tools:** Setup `ruff` (linting/formatting) and `pytest` configs inside `pyproject.toml`.
* **10.2 Configure `pre-commit`:** Check YAML formats, trailing whitespace, imports, and syntax.
* **10.3 GitHub Actions workflow (`.github/workflows/ci.yml`):**
* Linting and formatting checks (`ruff`)
* Deterministic test suite execution
* CPU smoke test execution


* **10.4 Dedicated benchmark workflow (`.github/workflows/benchmark.yml`):** Manual trigger for performance regression tracking.

### Acceptance Criteria

* CI runs cleanly on PRs and main branch commits without requiring GPU runners.

**Suggested Commits:**

```bash
chore: configure ruff pytest and development dependencies
chore: add pre-commit quality checks
ci: run linting formatting and tests on pull requests
ci: add manually triggered benchmark workflow

```

---

## 11. Phase 6: Structured Data Export

* **Goal:** Decouple match tracking data from visual overlays for independent analysis.
* **Estimated Effort:** 1–2 days

### Target Data Exports

| Export Level | Key Fields | Format |
| --- | --- | --- |
| **Player Tracking** | `frame_index`, `timestamp_seconds`, `player_id`, `team_id`, `x_image`, `y_image`, `x_pitch`, `y_pitch`, `speed_mps`, `distance_m`, `has_ball`, `detection_confidence`, `position_valid` | CSV / JSON |
| **Ball Tracking** | `frame_index`, `timestamp_seconds`, `x_image`, `y_image`, `x_pitch`, `y_pitch`, `detection_confidence`, `is_interpolated`, `assigned_player_id`, `assignment_distance` | CSV / JSON |
| **Possession Intervals** | `start_frame`, `end_frame`, `start_time`, `end_time`, `player_id`, `team_id`, `duration_seconds`, `confidence`, `termination_reason` | CSV / JSON |
| **Candidate Events** | `event_id`, `start_frame`, `end_frame`, `from_player_id`, `to_player_id`, `from_team_id`, `to_team_id`, `start_x`, `start_y`, `end_x`, `end_y`, `transition_frames`, `event_type`, `confidence` | CSV / JSON |

### Event Types

* `candidate_pass`
* `turnover`
* `recovery`
* `loose_ball`
* `uncertain_transition`
* `excluded`

### Acceptance Criteria

* Analytics metrics can be recalculated entirely from exported data files without re-running video frames.

**Suggested Commits:**

```bash
feat: define structured frame and possession event schemas
feat: export tracking and possession records as CSV and JSON
feat: record model configuration and run metadata
test: validate exported analytics schemas

```

---

## 12. Phase 7: Possession Heatmap

* **Goal:** Implement an explainable spatial density visualization of on-ball dominance.
* **Estimated Effort:** 2 days

### Pipeline Architecture

* **Input:** Qualified possession records on valid pitch coordinates (excluding unassigned or low-confidence states).
* **Coordinate Space:** Normalized (0.0 to 1.0) or metric pitch dimensions ($105\text{ m} \times 68\text{ m}$).
* **Duration Weighting:** Observation weight scaled by frame duration ($\Delta t$) to reflect true possession time instead of raw frame counts.
* **Filtering Capabilities:** Team ID, player Track ID, time/frame intervals, and minimum possession confidence thresholds.
* **Artifacts:**
* `outputs/report/heatmaps/team_1_possession.png`
* `outputs/report/heatmaps/team_2_possession.png`
* `outputs/report/heatmaps/player_<id>_possession.png`
* `outputs/report/data/possession_heatmap.csv`



### Acceptance Criteria

* Heatmaps measure actual on-ball possession duration rather than overall player positioning.

**Suggested Commits:**

```bash
feat: aggregate possession events in pitch coordinates
feat: render team and player possession heatmaps
feat: add time and confidence filters for heatmaps
test: cover heatmap aggregation and filtering
docs: explain possession heatmap methodology

```

---

## 13. Phase 8: Pass-Network Inference

* **Goal:** Infer team passing networks from tracked possession transitions while accounting for noise.
* **Estimated Effort:** 3–4 days

### Transition Logic & Network Formulation

* **Possession State Machine:** Temporal filtering with a hysteresis buffer (e.g., minimum 3 consecutive frames before assignment changes) to eliminate frame-level flicker.
* **Inference Categories:**
* `candidate_pass`: Controlled transition between teammates within a configured duration window.
* `turnover`: Transition from Player A to Opponent B.
* `recovery`: Transition from loose ball to stable possession.
* `uncertain_transition`: Transitions confounded by missing frames, track swaps, or off-pitch coordinates.


* **Graph Definition:**
* **Nodes:** Average pitch coordinates of track IDs during on-ball events; node radius scaled by possession duration.
* **Edges:** Directed connections scaled by candidate pass count; edge transparency mapped to average transition confidence.


* **Manual Validation:** Hand-label 20–50 transition intervals to produce precision/recall benchmarks for candidate passes and turnovers.

### Acceptance Criteria

* Node identities are clearly identified by Track ID (avoiding false real-player naming).
* Network outputs include candidate event classifications and validation precision/recall metrics.

**Suggested Commits:**

```bash
feat: stabilize frame-level possession assignments
feat: infer possession transition events
feat: classify candidate passes turnovers and recoveries
feat: calculate pass-network nodes and edges
feat: render track-level team pass networks
test: cover pass inference edge cases
docs: document pass inference assumptions and limitations

```

---

## 14. Phase 9: Benchmarking Framework

* **Goal:** Quantify runtime and detection trade-offs systematically across hardware and configurations.
* **Estimated Effort:** 3–5 days

### Benchmark Design

* **Target Configurations:** Model variants (`YOLOv8n`, `YOLOv8m`, `YOLOv8x`) swept across confidence levels ($0.10, 0.20, 0.30, 0.40, 0.50$).
* **Evaluation Scenarios (Short Clips):**
* *Clip A:* Standard wide-angle broadcast view
* *Clip B:* Dense player occlusion (box scrambles)
* *Clip C:* Rapid camera panning and zoom adjustments
* *Clip D:* Contrast issues (shadows, kit similarities)
* *Clip E:* Camera cuts and replays


* **Profiling Breakdown:** Video decoding, object detection inference, tracker update, camera motion estimation, perspective transformation, team assignment, possession logic, and frame rendering.
* **Metrics:** Latency (ms/stage), overall FPS, player detection precision/recall, ball detection precision/recall, track fragmentation, and possession stability.

### Acceptance Criteria

* Benchmarking script runs in an automated fashion and exports results to `reports/benchmarks/results.csv`.
* Every benchmark includes full hardware/environment metadata (`reports/benchmarks/environment.json`).

**Suggested Commits:**

```bash
bench: add reproducible benchmark runner
bench: record runtime and environment metadata
bench: compare supported model variants
bench: evaluate detection confidence thresholds
bench: profile end-to-end pipeline stages
docs: publish benchmark methodology and results

```

---

## 15. Phase 10: Rigorous Error Analysis

* **Goal:** Quantify and categorize failure modes with reproducible edge cases and mitigation steps.
* **Estimated Effort:** 2–3 days

### Failure Taxonomy

```text
Pipeline Failures
├── Detection
│   ├── Small-object ball miss
│   ├── False-positive ball detection
│   └── Referee misclassified as player
├── Tracking
│   ├── Track-ID switch
│   └── Re-identification loss through occlusion
├── Perspective & Motion
│   ├── Optical-flow drift from player movement
│   └── Non-planar pitch edge distortion
└── Analytics
    ├── Close-proximity possession flicker
    └── Track swap registered as candidate pass

```

### Mitigation Targets

| Identified Failure | Root Cause | Implemented / Proposed Mitigation |
| --- | --- | --- |
| **Ball Detection Dropout** | Motion blur / Small pixel footprint | Temporal interpolation window & class-specific confidence threshold |
| **Possession Flicker** | Cluttered player-ball distances | Temporal hysteresis buffer requiring $\ge N$ stable frames |
| **Camera Cut Breakdown** | Optical-flow accumulation failure | Scene-cut detector resetting homography & tracking state |
| **False Passing Edges** | Tracker identity switch on same player | Spatial displacement validation rejecting instantaneous transfers |

### Acceptance Criteria

* Curate a failure dataset of recorded clips with expected vs. observed outcomes.
* Implement and test at least one mitigation (e.g., possession temporal hysteresis), documenting the performance delta.

**Suggested Commits:**

```bash
analysis: define pipeline failure taxonomy
analysis: add reproducible failure case records
fix: stabilize possession across short detection gaps
fix: reject implausible pass transitions
docs: publish error analysis and mitigation results

```

---

## 16. Phase 11: README and Documentation Overhaul

* **Goal:** Deliver clear documentation allowing quick evaluation by engineers and recruiters.
* **Estimated Effort:** 1–2 days

### Documentation Checklist

* **Header & Overview:** Concise summary of technical capabilities and analytical features.
* **Visual Demo:** High-resolution GIF or video demo displaying bounding boxes, track trails, pitch radar, and active possession indicators.
* **Quickstart:** Clean, functional installation commands executed in 5 steps or fewer.
* **Pipeline Architecture:** ASCII or Mermaid diagram detailing data transformations from video input to analytics exports.
* **Engineering Standards:** Concrete coverage stats, CI status badges, and CLI references.
* **Analytical Limitations:** Explicit documentation covering camera angles, track swaps, and confidence assumptions.
* **Attribution:** Direct citations acknowledging the base tutorial and external models.

### Acceptance Criteria

* A new developer can set up and run the demo on sample video in under 10 minutes.

**Suggested Commits:**

```bash
docs: add architecture and methodology documentation
docs: add demo and analytics examples
docs: publish reproducible quick start
docs: document limitations benchmark and error analysis
docs: finalize attribution and original contribution summary

```

---

## 17. Phase 12: Final Quality Pass

* **Goal:** Verify end-to-end repository reproducibility from a clean system state.
* **Estimated Effort:** 1 day

### Execution Steps

1. **Fresh-Clone Test:** Execute a full setup inside an isolated container or fresh virtual environment:
```bash
git clone <repository-url>
cd football-cv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
football-cv validate --config configs/default.yaml
pytest
football-cv analyze --input examples/sample.mp4 --output outputs/demo.mp4
football-cv report --tracks outputs/tracks.json --output-dir outputs/report

```


2. **Link Integrity Audit:** Check all markdown links, image references, and asset download URLs.
3. **Repository Cleanup:** Remove orphan notebooks, temporary cache files, local absolute paths, and untracked artifacts.
4. **Secret Audit:** Scan git commit history for API keys, local credentials, or hardcoded sensitive parameters.
5. **Release Tagging:** Tag and publish release `v1.0.0` accompanied by validation reports and sample figures.

### Acceptance Criteria

* Zero failures across fresh clone, test suite, and execution commands.
* Release `v1.0.0` published with attached artifacts.

**Suggested Commits:**

```bash
chore: remove obsolete files and development artifacts
fix: resolve fresh-install reproducibility issues
docs: verify commands links and asset instructions
chore: prepare first reproducible release

```

---

## 18. Branch and Pull Request Strategy

```text
main
├── docs/project-attribution-and-metadata      (PR 1)
├── refactor/config-and-cli                    (PR 2)
├── refactor/python-package                    (PR 3)
├── test/core-logic-and-ci                     (PR 4)
├── feat/analytics-event-export                (PR 5)
├── feat/possession-heatmap                    (PR 6)
├── feat/pass-network                          (PR 7)
├── analysis/model-benchmark                   (PR 8)
├── analysis/failure-analysis                  (PR 9)
└── docs/portfolio-release                     (PR 10)

```

Each PR must include:

* Motivation and architectural summary
* Direct evidence of test coverage
* Before-and-after performance/output comparisons

---

## 19. Realistic Timeline

```text
Week 1: Engineering Foundation
├── Day 1: Baseline capture, attribution, and license verification
├── Day 2: Configuration models and validation rules
├── Day 3: CLI implementation and path decoupling
├── Day 4: Package migration under src/football_cv
└── Day 5: Orchestration logic, structured logging, output verification

Week 2: Reliability & Data Foundation
├── Day 6: Configuration and geometry unit tests
├── Day 7: Possession and interpolation tests
├── Day 8: Perspective transformation and kinematic tests
├── Day 9: Integration smoke test and ruff configuration
└── Day 10: CI automation and structured JSON/CSV data export

Week 3: Advanced Analytics Layer
├── Day 11: Possession state machine and temporal stabilization
├── Day 12: Pitch spatial mapping and heatmap aggregation
├── Day 13: Heatmap visualization filters and exports
├── Day 14: Pass-event inference state machine
└── Day 15: Directed network graphs and edge-weighting generation

Week 4: Empirical Evaluation
├── Day 16: Test clip dataset selection and ground truth annotation
├── Day 17: Model variant and confidence threshold sweeps
├── Day 18: Stage-level profiling and throughput evaluation
├── Day 19: Failure case cataloging and root-cause analysis
└── Day 20: Mitigation engineering and re-benchmarking

Week 5: Release & Portfolio Finalization
├── Day 21: Architecture, pipeline, and methodology documentation
├── Day 22: High-resolution demo creation and figure exports
├── Day 23: Benchmark and error-analysis reports
├── Day 24: Isolated fresh-clone validation
└── Day 25: Release v1.0.0 publishing and interview talking points

```

---

## 20. Priority Matrix

```text
High Impact │  [P0] Reproducible CLI & Package      [P0] Possession Heatmaps
            │  [P0] Event Data Export (CSV/JSON)    [P0] Pass-Network Inference
            │  [P0] Testing & CI Pipeline           [P0] Empirical Benchmark Suite
            │  [P0] Attribution & Licensing         [P0] Error & Failure Taxonomy
            ├───────────────────────────────────────────────────────────────────
            │  [P1] Pre-commit Hooks                [P2] Parquet Support
            │  [P1] Possession Hysteresis Fix       [P2] Multi-configuration Profiles
            │  [P1] Camera Cut Recovery Logic       [P2] Issue Templates & Web Docs
 Low Impact │  [P1] Runtime Profiling Plots         [P2] Additional Vision Backbones
            └───────────────────────────────────────────────────────────────────
               Low Complexity                       High Complexity

```

*Complete all P0 requirements before addressing P1 and P2 items.*

---

## 21. Definition of Done

* **Engineering:** Clean package installation via `pyproject.toml`; unified CLI execution; structured logging; passing unit and integration tests in CI.
* **Transparency:** Clear attribution documentation; licenses documented for models and sample footage.
* **Analytics:** Structured possession intervals, candidate passes, and heatmaps exported from pipeline data.
* **Evaluation:** Benchmarking reports covering speed/accuracy trade-offs; documented failure taxonomy with at least one targeted mitigation.
* **Presentation:** Clean README with demo clips, pipeline architecture, clear setup steps, and release `v1.0.0`.

---

## 22. Recruiter & Technical Interview Narrative

> *"I began with a tutorial-based football tracking pipeline to examine baseline methods for object detection, tracking, optical flow, and pitch homography. I refactored the codebase into a tested, configurable Python package, added structured event exports, built possession heatmaps and pass-network inference, and benchmarked the pipeline across model variants and challenging broadcast conditions. Finally, I documented system failure modes and implemented targeted mitigations for possession stability."*

This positions your work around engineering discipline, machine learning evaluation, and practical sports analytics.