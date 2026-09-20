# Football Vision Engine: C++ & Hybrid Systems Transformation Plan

> **Strategic Objective:** Transform `football-cv` from a Python computer-vision prototype into a production-grade, hybrid ML systems project. The finished repository demonstrates modern C++, OpenCV, Python/C++ interoperability via pybind11, quantitative profiling, strict numerical parity testing, ONNX Runtime model deployment, and engineering judgment.

---

## 1. Executive Summary & Core Philosophy

### Target Story
> *"I built a football video analytics pipeline, profiled it to identify latency bottlenecks, moved performance-sensitive vision and geometry components into a tested C++ core, integrated that core with Python using pybind11, deployed YOLO inference through ONNX Runtime, and quantitatively evaluated both system throughput and detection/tracking fidelity."*

### Key Architectural Principle
**Python owns orchestration, configuration, experimentation, and analytics.**  
**C++ owns performance-critical systems, geometric transformations, and computer vision routines.**

```
                     VIDEO INPUT
                          │
                          ▼
               ┌─────────────────────┐
               │ Video Decode (CV2)  │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Object Detection   │
               │ PyTorch / ONNX C++  │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Multi-Object Track  │
               │      ByteTrack      │
               └──────────┬──────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│     C++ Vision Core     │     │    Python Analytics     │
│   (football_cv_core)    │     │      (football_cv)      │
│  • BoundingBox Geometry │     │  • Team Classification  │
│  • Metric Homography    │     │  • Ball Possession      │
│  • Optical Flow Motion  │     │  • Tactical Aggregation │
│  • Coordinate Filtering │     │  • Visualization / PDF  │
└────────────┬────────────┘     └────────────┬────────────┘
             │                               │
             └────────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────┐
               │    Output Analytics     │
               │ • Transformed Metrics   │
               │ • Speed & Distance (m)  │
               │ • Annotated Video & CSV │
               └─────────────────────────┘
```

### Things We Deliberately DO NOT Do
1. ❌ **Do NOT rewrite everything in C++:** Python is the industry standard for high-level ML orchestration.
2. ❌ **Do NOT implement custom matrix/linear algebra math:** Use OpenCV's tested C++ primitives.
3. ❌ **Do NOT rewrite YOLO or ByteTrack from scratch:** Integrate established architectures via ONNX Runtime and C++ APIs.
4. ❌ **Do NOT jump immediately to CUDA:** Establish CPU baselines, profile, migrate to C++, deploy ONNX, and *only* introduce CUDA if profiling justifies it.
5. ❌ **Do NOT fabricate optimization stories:** If a C++ component provides only marginal speedup because OpenCV Python bindings already call native C, document that insight. That proves senior engineering maturity.

---

## 2. Baseline Assessment (Current Repository State)

| Component | Status | Details |
|---|---|---|
| **Python Architecture** | ✅ Completed | Clean package layout in `src/football_cv/` with modular subpackages (`tracking`, `camera_motion`, `perspective`, `movement`, `teams`, `possession`, `rendering`, `analytics`). |
| **Test Suite** | ✅ Active | 139 unit and integration tests passing (`pytest` in `.venv`). |
| **Stage Profiling** | 🟡 Existing Foundation | `src/football_cv/benchmark/runner.py` provides initial stage-level timing via `time_stage`. Needs frozen baseline artifact. |
| **C++ Build Environment** | ✅ Ready | Microsoft Visual Studio 2022 BuildTools (MSVC `cl.exe`) and bundled `cmake.exe` available. |
| **Native C++ Library** | ⏳ Not Started | Target: `cpp/` directory with `CMakeLists.txt`, `football_cv_core`, and unit tests. |
| **Python Bindings** | ⏳ Not Started | Target: `pybind11` bridge with zero-copy NumPy interoperability. |
| **ONNX Runtime Engine** | ⏳ Not Started | Target: YOLO export to ONNX + C++ ONNX Runtime inference backend. |

---

## 3. Target Repository Structure

```text
football-cv/
├── .github/
│   └── workflows/
│       ├── python-ci.yml           # Pytest, ruff lint & format
│       └── cpp-ci.yml              # CMake build, CTest, clang-format
│
├── benchmarks/
│   ├── data/                       # Standard test clips (short, diverse motion)
│   ├── baseline_results.json       # Pure Python reference benchmarks
│   ├── hybrid_results.json         # Python + C++ core benchmarks
│   └── onnx_results.json           # ONNX Runtime benchmarks
│
├── configs/
│   ├── default.yaml                # Standard pipeline configuration
│   └── benchmarks.yaml             # Profiling and backend toggle configs
│
├── cpp/                            # Standalone C++ Vision Core (football_cv_core)
│   ├── CMakeLists.txt              # Modern CMake configuration (C++17/20)
│   ├── include/
│   │   └── football_cv/
│   │       ├── types.hpp           # Point2D, BoundingBox, Detection structs
│   │       ├── geometry.hpp        # BBox operations, Euclidean distances
│   │       ├── perspective.hpp     # 4-point homography & pointPolygonTest
│   │       ├── camera_motion.hpp   # Lucas-Kanade optical flow compensation
│   │       └── inference/
│   │           ├── detector.hpp    # Abstract detector interface
│   │           └── onnx_detector.hpp # ONNX Runtime C++ backend
│   ├── src/
│   │   ├── geometry.cpp
│   │   ├── perspective.cpp
│   │   ├── camera_motion.cpp
│   │   └── inference/
│   │       └── onnx_detector.cpp
│   ├── bindings/
│   │   └── python_module.cpp       # pybind11 module definition
│   └── tests/
│       ├── CMakeLists.txt          # CTest test runner setup (GoogleTest / Catch2)
│       ├── test_geometry.cpp
│       ├── test_perspective.cpp
│       └── test_camera_motion.cpp
│
├── docs/
│   ├── cpp_transformation_plan.md  # THIS ROADMAP
│   ├── architecture.md             # Hybrid architecture documentation
│   ├── benchmarks.md               # Empirical latency & throughput reports
│   └── design_decisions.md         # Rationale for language & library choices
│
├── src/
│   └── football_cv/                # Python package orchestration
│       ├── core/                   # C++ module import shim / fallback
│       │   ├── __init__.py         # Imports from native extension if built
│       │   └── backend.py          # Selector: 'cpp' vs 'python'
│       ├── camera_motion/
│       ├── perspective/
│       ├── tracking/
│       ├── utils/
│       └── ...
│
├── tests/
│   ├── unit/                       # Existing Python unit tests
│   ├── integration/                # Pipeline integration tests
│   └── parity/
│       └── test_cpp_parity.py      # Strict numerical parity: Python vs C++
│
├── CMakeLists.txt                  # Top-level CMake (orchestrates cpp/ & bindings)
├── pyproject.toml                  # Python package metadata + setuptools/scikit-build
└── README.md                       # Portfolio-grade presentation with metrics & GIF
```

---

## 4. Phase-by-Phase Execution Plan

### Phase 0: Establish & Freeze Baseline
*Goal: Measure the existing system thoroughly before changing or rewriting anything.*

1. **Benchmark Footage Definition:**
   - Select 3 short reference video clips (30–60s) with varied motion dynamics:
     - `clip_static_camera.mp4`: Fixed angle, minimal pan.
     - `clip_pan.mp4`: Continuous horizontal panning along touchline.
     - `clip_complex.mp4`: Rapid cuts, zooming, high player density.
2. **Comprehensive Timing Instrumentation:**
   - Expand `src/football_cv/benchmark/runner.py` to isolate:
     1. Video decode / I/O latency
     2. YOLO detection latency
     3. ByteTrack update latency
     4. Camera motion estimation (optical flow) latency
     5. Perspective transformation latency
     6. Speed and distance calculation latency
     7. Team assignment and possession latency
     8. Rendering / visualization latency
     9. Total end-to-end frame latency and mean FPS
3. **Artifact Generation:**
   - Run benchmark on standardized hardware.
   - Produce `benchmarks/baseline_results.json` capturing hardware specs, OS, compiler, frame counts, per-component latencies, and FPS.
   - **Key Deliverable Question:** Where is the runtime actually being spent?

---

### Phase 1: Python Architecture Validation & Abstraction
*Goal: Ensure clean module boundaries and backend interfaces exist before introducing C++.*

1. **Verify Interface Contracts:**
   - Ensure geometric functions in [`src/football_cv/utils/geometry.py`](file:///d:/football-cv/src/football_cv/utils/geometry.py) have deterministic, typed signatures.
   - Ensure [`PerspectiveTransformer`](file:///d:/football-cv/src/football_cv/perspective/transformer.py) accepts array-like inputs and returns well-formed pitch coordinates.
   - Ensure [`CameraMotionEstimator`](file:///d:/football-cv/src/football_cv/camera_motion/estimator.py) accepts frame sequences and returns `(dx, dy)` translation tuples.
2. **Configuration Backend Flag:**
   - Add `vision.backend: "python" | "cpp"` to `config.yaml` to allow toggling backends at runtime for testing and benchmarking.

---

### Phase 2: Standalone C++ Vision Core (`football_cv_core`)
*Goal: Build a tested, clean C++ library with modern CMake before touching Python bindings.*

Build command standard:
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release
ctest --test-dir cpp/build --output-on-failure
```

#### Component A: Geometry Library
- **Headers:** `cpp/include/football_cv/types.hpp`, `cpp/include/football_cv/geometry.hpp`
- **Source:** `cpp/src/geometry.cpp`
- **Data Structures:**
  ```cpp
  namespace football_cv {
  struct Point2D {
      double x{0.0};
      double y{0.0};
      double distance_to(const Point2D& other) const;
  };

  struct BoundingBox {
      double x1{0.0};
      double y1{0.0};
      double x2{0.0};
      double y2{0.0};
      Point2D center() const;
      Point2D foot_position() const;
      double width() const;
      double height() const;
      bool is_valid() const;
  };
  }
  ```
- **Tests:** `cpp/tests/test_geometry.cpp` (verifying centers, foot positions, zero-area boxes, negative coordinates, NaNs).

#### Component B: Perspective Transformer
- **Headers:** `cpp/include/football_cv/perspective.hpp`
- **Source:** `cpp/src/perspective.cpp`
- **Implementation:**
  - Encapsulate `cv::Mat homography_` computed via `cv::getPerspectiveTransform`.
  - Check boundary containment with `cv::pointPolygonTest`.
  - Transform coordinate points with `cv::perspectiveTransform`.
- **Tests:** `cpp/tests/test_perspective.cpp` (verifying known quad-to-pitch mappings and points outside boundary returning nullopt/empty).

#### Component C: Camera Motion Estimator
- **Headers:** `cpp/include/football_cv/camera_motion.hpp`
- **Source:** `cpp/src/camera_motion.cpp`
- **Implementation:**
  - Lucas-Kanade optical flow using `cv::calcOpticalFlowPyrLK` and `cv::goodFeaturesToTrack`.
  - Vertical border masking to exclude players on the pitch.
  - Scene cut threshold detection for sudden optical flow jumps.
- **Tests:** `cpp/tests/test_camera_motion.cpp` (synthetic synthetic translation shifts, identical frames, scene cuts).

---

### Phase 3: Python/C++ Interoperability via pybind11
*Goal: Expose the compiled C++ core to Python with seamless NumPy array interoperability.*

1. **pybind11 CMake Setup:**
   - Link `pybind11::module` in `cpp/CMakeLists.txt`.
   - Expose module named `_core` (or `football_cv_core`).
2. **Type Conversions & Zero-Copy Semantics:**
   - Map `cv::Mat` to/from `py::array_t<uint8_t>` without unnecessary memory allocations.
   - Map `std::vector<Point2D>` to/from NumPy 2D float arrays.
3. **Python Facade Integration:**
   - Create `src/football_cv/core/`:
     ```python
     try:
         from football_cv import _core as native_core

         HAS_CPP_CORE = True
     except ImportError:
         native_core = None
         HAS_CPP_CORE = False
     ```
   - When `config.vision.backend == "cpp"`, delegate calls to `native_core`.

---

### Phase 4: Numerical Parity Validation
*Goal: Prove mathematically that the C++ port matches the reference Python implementation.*

1. **Parity Test Suite (`tests/parity/test_cpp_parity.py`):**
   - **BBox calculations:** Center and foot coordinates match exactly.
   - **Homography:** Given identical pixel vertices and test points:
     ```python
     np.testing.assert_allclose(py_pts, cpp_pts, rtol=1e-5, atol=1e-6)
     ```
   - **Optical flow:** Compare camera translation vectors over 50 real match frames.
   - **Edge cases:** Empty lists, points outside polygons, zero-sized boxes handled identically.

---

### Phase 5: Empirical Benchmarking & Profiling
*Goal: Produce quantitative comparisons and document the engineering reality.*

1. **Side-by-Side Matrix:**
   - Run benchmark on identical hardware:
     - Pure Python
     - Hybrid Python + C++ Core
2. **Detailed Component Breakdown:**
   | Component | Python Latency | C++ Latency | Speedup | Rationale |
   |---|---|---|---|---|
   | Geometry / BBox Ops | X ms | Y ms | Zx | Avoids Python object overhead |
   | Perspective Transform | X ms | Y ms | Zx | Comparison of py cv2 vs native C++ |
   | Optical Flow Compensation | X ms | Y ms | Zx | Python cv2 wrapper vs C++ loop overhead |
   | Pipeline Total FPS | X FPS | Y FPS | Z% | End-to-end impact assessment |
3. **Engineering Analysis in `docs/benchmarks.md`:**
   - Explain why certain operations improve dramatically while others show modest gains (e.g., OpenCV's Python bindings already call native C++ routines).

---

### Phase 6: ONNX Runtime Model Deployment (Stretch Milestone)
*Goal: Address the primary pipeline bottleneck (model inference) with an abstract C++ inference engine.*

1. **Export YOLO to ONNX:**
   - Script: `scripts/export_yolo_onnx.py`.
   - Validate ONNX graph integrity with `onnx.checker.check_model`.
2. **Abstract Detector Architecture:**
   ```cpp
   namespace football_cv {
   class Detector {
   public:
       virtual ~Detector() = default;
       virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
   };
   }
   ```
3. **ONNX Runtime C++ Detector:**
   - Header: `cpp/include/football_cv/inference/onnx_detector.hpp`.
   - Source: `cpp/src/inference/onnx_detector.cpp`.
   - Implement Letterbox preprocessing, tensor allocation, Ort::Session execution, Non-Maximum Suppression (NMS), and coordinate scaling.
4. **Benchmarking Inference:**
   - Compare PyTorch CUDA/CPU vs ONNX Runtime CPU vs ONNX Runtime TensorRT/DirectML.

---

### Phase 7: GitHub Actions CI Matrix
*Goal: Ensure cross-platform automated testing and code style enforcement.*

1. **Python CI:**
   - Pytest execution across Python 3.10 and 3.11.
   - Ruff linting and formatting verification.
2. **C++ CI:**
   - CMake configuration and build on `ubuntu-latest` and `windows-latest`.
   - CTest execution for all native unit tests.
   - Clang-format style check (`clang-format --dry-run --Werror`).
3. **Integration CI:**
   - Build native extension with pybind11 and execute numerical parity tests.

---

### Phase 8: Portfolio Presentation & Interview Story
*Goal: Present the project with recruiter-optimized visual assets and technical depth.*

1. **README Structure:**
   - **One-sentence hook:** *"Football Vision Engine: Real-time football analytics with Python, modern C++, OpenCV, ByteTrack, and ONNX Runtime."*
   - **High-impact Demo GIF / Video clip** (first 5 seconds show tracked players, speed metrics, team colors, mini-map homography).
   - **Architecture Diagram** (Python orchestration + C++ native core).
   - **Empirical Performance Table** (Python baseline vs Hybrid vs ONNX).
   - **Design Decisions & Trade-offs** (Why pybind11? Why not rewrite everything? What did profiling reveal?).
2. **Resume Checkpoints:**
   - **Checkpoint 1 (Milestone 2):** Developed C++/CMake vision backend for geometry and optical flow.
   - **Checkpoint 2 (Milestone 3 & 4):** Built hybrid Python/C++ pipeline with pybind11, automated numerical parity testing, and component-level profiling.
   - **Checkpoint 3 (Milestone 5 & 6):** Deployed ONNX Runtime C++ inference, achieving verified latency reduction and documented systems trade-offs.

---

## 5. C++ Curriculum & Learning Objectives

This project serves as Donald's hands-on modern C++ curriculum:

| Topic Area | Specific Concepts to Practice |
|---|---|
| **Core C++** | Stack vs. heap, references vs. pointers, `const` correctness, RAII, rule of 0/3/5, namespaces, `enum class`. |
| **Modern C++ (17/20)** | `std::optional`, `std::unique_ptr`, `std::string_view`, structured bindings, lambdas, range-based loops. |
| **Build Engineering** | Modern target-based CMake (`target_include_directories`, `target_link_libraries`), Release vs. Debug configurations, compiler warning flags (`/W4` on MSVC, `-Wall -Wextra` on GCC/Clang). |
| **Computer Vision (OpenCV)** | `cv::Mat` memory layout, continuous vs. non-continuous strides, BGR channel ordering, avoiding unnecessary deep copies, matrix ROI slicing. |
| **Interoperability (pybind11)** | Exposing classes and structs, binding STL containers (`std::vector`), buffer protocol and NumPy array wrapping (`py::array_t`), managing GIL states during compute-heavy loops. |

---

## 6. Commit Convention Discipline

To reflect software engineering maturity in GitHub history:

| Prefix | Scope | Example |
|---|---|---|
| `bench:` | Baseline & profiling | `bench: add component-level pipeline latency instrumentation` |
| `build:` | CMake & tooling | `build: initialize CMake configuration for football_cv_core` |
| `feat(cpp):` | C++ implementation | `feat(cpp): implement BoundingBox and Point2D geometry utilities` |
| `test(cpp):` | C++ unit tests | `test(cpp): add unit tests for perspective homography transformer` |
| `feat(bindings):` | pybind11 bridge | `feat(bindings): expose C++ geometry and homography core via pybind11` |
| `test(parity):` | Equivalence tests | `test(parity): verify Python and C++ numerical parity on match frames` |
| `feat(inference):`| ONNX Runtime | `feat(inference): integrate C++ ONNX Runtime detector backend` |
| `ci:` | GitHub Actions | `ci: add matrix build workflow for CMake and CTest` |
| `docs:` | Documentation | `docs: add empirical benchmark results and design rationale` |

---

## 7. Immediate Next Steps

When ready to begin execution:
1. **Freeze Baseline:** Run and record initial profiling figures using `src/football_cv/benchmark/runner.py` on a standard clip to produce `benchmarks/baseline_results.json`.
2. **Scaffold `cpp/` Directory:** Create initial `cpp/CMakeLists.txt` and verify compilation using MSVC / CMake on the local Windows machine.
3. **Implement Component A (Geometry):** Write `cpp/include/football_cv/geometry.hpp`, `cpp/src/geometry.cpp`, and `cpp/tests/test_geometry.cpp`.
