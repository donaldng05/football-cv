"""
High-resolution micro-benchmarking suite for core mathematical and vision kernels.

Measures isolated CPU execution latency and throughput across:
1. Geometry and bounding box operations (Python vs C++ native structs)
2. Perspective transformation and polygon boundary checks (Python cv2 vs C++ native homography)
3. Camera motion feature displacement and scene-cut evaluation (Python zip loop vs C++ native loop)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core import has_cpp_core
from ..core.backend import (
    CppPerspectiveTransformerAdapter,
)
from ..exceptions import BenchmarkError
from ..perspective.transformer import PerspectiveTransformer as PyPerspectiveTransformer
from ..utils.geometry import (
    extract_centers as py_extract_centers,
)
from ..utils.geometry import (
    filter_valid_bboxes as py_filter_valid_bboxes,
)
from ..utils.geometry import (
    find_nearest_point as py_find_nearest_point,
)
from ..utils.geometry import (
    get_center_of_bbox as py_get_center_of_bbox,
)
from ..utils.geometry import (
    get_foot_position as py_get_foot_position,
)
from ..utils.geometry import (
    measure_distance as py_measure_distance,
)
from ..utils.geometry import (
    measure_xy_distance as py_measure_xy_distance,
)
from .environment import EnvironmentCollector

logger = logging.getLogger(__name__)

if has_cpp_core():
    from .. import _core as native_core
else:
    native_core = None


@dataclass
class MicroBenchmarkResult:
    """Quantitative measurement record for an isolated micro-benchmark kernel."""

    name: str
    category: str
    iterations: int
    python_latency_us: float
    cpp_latency_us: float
    speedup: float
    python_throughput_ops: float
    cpp_throughput_ops: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_callable(fn: Any, iterations: int, warmup: int = 50) -> tuple[float, float]:
    """
    Measure execution time of a zero-argument callable using nanosecond counters.

    Returns:
        tuple of (mean_latency_microseconds, throughput_ops_per_second)
    """
    for _ in range(warmup):
        fn()

    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    t1 = time.perf_counter_ns()

    total_ns = max(1, t1 - t0)
    total_sec = total_ns / 1e9
    mean_latency_us = (total_ns / iterations) / 1000.0
    throughput = iterations / total_sec
    return round(mean_latency_us, 4), round(throughput, 1)


def benchmark_geometry(
    iterations: int = 100_000,
    batch_size: int = 25,
) -> list[MicroBenchmarkResult]:
    """
    Benchmark geometric calculations and bounding box metrics.
    """
    if native_core is None:
        raise BenchmarkError(
            "Native C++ vision core is required for micro-benchmarking."
        )

    results: list[MicroBenchmarkResult] = []

    # 1. Bounding box center
    py_box = [150.0, 200.0, 350.0, 500.0]
    cpp_box = native_core.BoundingBox(150.0, 200.0, 350.0, 500.0)

    py_lat, py_tp = _time_callable(lambda: py_get_center_of_bbox(py_box), iterations)
    cpp_lat, cpp_tp = _time_callable(lambda: cpp_box.center(), iterations)
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="bbox_center",
            category="Geometry",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="Bounding box center calculation ([x1,y1,x2,y2] -> Point2D)",
        )
    )

    # 2. Bounding box foot position
    py_lat, py_tp = _time_callable(lambda: py_get_foot_position(py_box), iterations)
    cpp_lat, cpp_tp = _time_callable(lambda: cpp_box.foot_position(), iterations)
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="bbox_foot_position",
            category="Geometry",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="Player foot position ([x1,y1,x2,y2] -> bottom-center Point2D)",
        )
    )

    # 3. Euclidean distance
    p1_py = (120.5, 340.2)
    p2_py = (450.8, 780.4)
    p1_cpp = native_core.Point2D(120.5, 340.2)
    p2_cpp = native_core.Point2D(450.8, 780.4)

    py_lat, py_tp = _time_callable(
        lambda: py_measure_distance(p1_py, p2_py), iterations
    )
    cpp_lat, cpp_tp = _time_callable(lambda: p1_cpp.distance_to(p2_cpp), iterations)
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="euclidean_distance",
            category="Geometry",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="Euclidean distance between two 2D points",
        )
    )

    # 4. Batch centers extraction (e.g. 25 bboxes per frame)
    batch_py = [
        [float(i * 10), float(i * 15), float(i * 10 + 50), float(i * 15 + 80)]
        for i in range(batch_size)
    ]
    batch_cpp = [native_core.BoundingBox(b[0], b[1], b[2], b[3]) for b in batch_py]
    batch_iters = max(100, iterations // 10)

    py_lat, py_tp = _time_callable(lambda: py_extract_centers(batch_py), batch_iters)
    cpp_lat, cpp_tp = _time_callable(
        lambda: native_core.extract_centers(batch_cpp), batch_iters
    )
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name=f"batch_centers_{batch_size}x",
            category="Geometry",
            iterations=batch_iters,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description=f"Batch center extraction across {batch_size} tracking bboxes",
        )
    )

    # 5. Filter valid bboxes
    mixed_py = [*batch_py, [100.0, 200.0, 50.0, 150.0], [0.0, 0.0, 0.0, 0.0]]
    mixed_cpp = [native_core.BoundingBox(b[0], b[1], b[2], b[3]) for b in mixed_py]

    py_lat, py_tp = _time_callable(
        lambda: py_filter_valid_bboxes(mixed_py), batch_iters
    )
    cpp_lat, cpp_tp = _time_callable(
        lambda: native_core.filter_valid_bboxes(mixed_cpp), batch_iters
    )
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="filter_valid_bboxes",
            category="Geometry",
            iterations=batch_iters,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description=f"Filter and validate {len(mixed_py)} bounding boxes (positive area)",
        )
    )

    # 6. Nearest neighbor search (22 players on pitch)
    target_py = (500.0, 400.0)
    candidates_py = [(float(i * 45), float(i * 35)) for i in range(22)]
    target_cpp = native_core.Point2D(500.0, 400.0)
    candidates_cpp = [native_core.Point2D(c[0], c[1]) for c in candidates_py]

    py_lat, py_tp = _time_callable(
        lambda: py_find_nearest_point(target_py, candidates_py, max_distance=300.0),
        batch_iters,
    )
    cpp_lat, cpp_tp = _time_callable(
        lambda: native_core.find_nearest_point(
            target_cpp, candidates_cpp, max_distance=300.0
        ),
        batch_iters,
    )
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="find_nearest_point_22x",
            category="Geometry",
            iterations=batch_iters,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="Find nearest player coordinate among 22 candidates within radius",
        )
    )

    return results


def benchmark_perspective(
    iterations: int = 20_000,
    batch_points: int = 100,
) -> list[MicroBenchmarkResult]:
    """
    Benchmark perspective transformation and pitch boundary containment tests.
    """
    if native_core is None:
        raise BenchmarkError(
            "Native C++ vision core is required for micro-benchmarking."
        )

    results: list[MicroBenchmarkResult] = []

    pixel_vertices = [[110.0, 1035.0], [265.0, 275.0], [910.0, 260.0], [1640.0, 915.0]]
    court_width = 68.0
    court_length = 23.32

    py_transformer = PyPerspectiveTransformer(pixel_vertices, court_width, court_length)
    cpp_adapter = CppPerspectiveTransformerAdapter(
        pixel_vertices, court_width, court_length
    )
    cpp_core = native_core.PerspectiveTransformer(
        pixel_vertices, court_width, court_length
    )

    test_pt = (500.0, 500.0)
    cpp_pt = native_core.Point2D(500.0, 500.0)

    # 1. Single-point perspective transform (including polygon test)
    py_lat, py_tp = _time_callable(
        lambda: py_transformer.transform_point(test_pt), iterations
    )
    cpp_lat, cpp_tp = _time_callable(
        lambda: cpp_adapter.transform_point(test_pt), iterations
    )
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="perspective_transform_point",
            category="Perspective",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="Single-point projection from broadcast pixel to metric pitch (with adapter)",
        )
    )

    # 2. Pure native single-point transform (bypassing NumPy array wrapping)
    raw_lat, raw_tp = _time_callable(
        lambda: cpp_core.transform_point(500.0, 500.0), iterations
    )
    raw_speedup = round(py_lat / max(1e-6, raw_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="perspective_transform_native_raw",
            category="Perspective",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=raw_lat,
            speedup=raw_speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=raw_tp,
            description="Pure native transform_point(x, y) without NumPy array conversion",
        )
    )

    # 3. Boundary polygon containment check
    poly_pts = np.array(pixel_vertices, dtype=np.float32)
    py_lat, py_tp = _time_callable(
        lambda: cv2.pointPolygonTest(poly_pts, (500, 500), False) >= 0,
        iterations,
    )
    cpp_lat, cpp_tp = _time_callable(
        lambda: cpp_core.is_point_inside(cpp_pt), iterations
    )
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name="polygon_containment_check",
            category="Perspective",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="2D Point-in-polygon containment test for pitch boundaries",
        )
    )

    # 4. Batch points transform (100 points)
    rng = np.random.RandomState(42)
    pts_x = rng.uniform(300.0, 900.0, size=batch_points)
    pts_y = rng.uniform(300.0, 800.0, size=batch_points)
    batch_pts_py = list(zip(pts_x, pts_y, strict=False))
    batch_pts_cpp = [native_core.Point2D(x, y) for x, y in batch_pts_py]
    batch_iters = max(100, iterations // 10)

    py_lat, py_tp = _time_callable(
        lambda: [py_transformer.transform_point(p) for p in batch_pts_py],
        batch_iters,
    )
    cpp_lat, cpp_tp = _time_callable(
        lambda: cpp_core.transform_points(batch_pts_cpp),
        batch_iters,
    )
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)
    results.append(
        MicroBenchmarkResult(
            name=f"batch_perspective_{batch_points}pts",
            category="Perspective",
            iterations=batch_iters,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description=f"Batch perspective projection of {batch_points} candidate points",
        )
    )

    return results


def benchmark_camera_motion(
    iterations: int = 2_000,
    num_features: int = 100,
) -> list[MicroBenchmarkResult]:
    """
    Benchmark optical flow feature loop, displacement vector computation, and scene-cut logic.
    """
    if native_core is None:
        raise BenchmarkError(
            "Native C++ vision core is required for micro-benchmarking."
        )

    results: list[MicroBenchmarkResult] = []

    rng = np.random.RandomState(42)
    # Synthetic optical flow features with 3.5 px shift
    good_old = rng.uniform(50.0, 900.0, size=(num_features, 2)).astype(np.float32)
    good_new = (
        good_old
        + np.array([3.2, 1.4], dtype=np.float32)
        + rng.normal(0, 0.2, size=(num_features, 2)).astype(np.float32)
    )

    min_dist = 5.0
    scene_cut = 80.0
    cpp_cam_core = native_core.CameraMotionEstimator(min_dist, scene_cut)

    # Pure Python displacement loop
    def _py_feature_loop() -> tuple[float, float, bool]:
        max_distance = 0.0
        cam_dx, cam_dy = 0.0, 0.0
        for new, old in zip(good_new, good_old, strict=False):
            dist = py_measure_distance(new.ravel(), old.ravel())
            if dist > max_distance:
                max_distance = dist
                cam_dx, cam_dy = py_measure_xy_distance(old.ravel(), new.ravel())

        is_cut = max_distance > scene_cut
        if is_cut or max_distance <= min_dist:
            return 0.0, 0.0, is_cut
        return cam_dx, cam_dy, is_cut

    # C++ native displacement calculation
    def _cpp_feature_loop() -> tuple[float, float, bool]:
        motion = cpp_cam_core.estimate_from_features(good_old, good_new)
        return motion.dx, motion.dy, motion.is_scene_cut

    py_lat, py_tp = _time_callable(_py_feature_loop, iterations)
    cpp_lat, cpp_tp = _time_callable(_cpp_feature_loop, iterations)
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)

    results.append(
        MicroBenchmarkResult(
            name=f"optical_flow_feature_loop_{num_features}pts",
            category="Camera Motion",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description=(
                f"Feature loop over {num_features} optical flow points (max displacement, "
                "sub-threshold filtering, and scene cut detection)"
            ),
        )
    )

    # Margin filtering benchmark
    all_pts = rng.uniform(0.0, 1080.0, size=(200, 2)).astype(np.float32)

    def _py_margin_filter() -> list[tuple[float, float]]:
        filtered = []
        for pt in all_pts:
            x = pt[0]
            if x <= 20.0 or (900.0 <= x <= 1050.0):
                filtered.append((float(pt[0]), float(pt[1])))
        return filtered

    def _cpp_margin_filter() -> list[Any]:
        return native_core.CameraMotionEstimator.filter_margin_points(all_pts, 1920.0)

    py_lat, py_tp = _time_callable(_py_margin_filter, iterations)
    cpp_lat, cpp_tp = _time_callable(_cpp_margin_filter, iterations)
    speedup = round(py_lat / max(1e-6, cpp_lat), 2)

    results.append(
        MicroBenchmarkResult(
            name="camera_motion_margin_filter_200pts",
            category="Camera Motion",
            iterations=iterations,
            python_latency_us=py_lat,
            cpp_latency_us=cpp_lat,
            speedup=speedup,
            python_throughput_ops=py_tp,
            cpp_throughput_ops=cpp_tp,
            description="Exclusion filter isolating vertical border margins across 200 keypoints",
        )
    )

    return results


def run_all_micro_benchmarks(scale: float = 1.0) -> dict[str, Any]:
    """
    Execute full micro-benchmark suite across all kernel categories.

    Args:
        scale: Scaling factor for iterations (1.0 for production, 0.01 for fast unit tests).

    Returns:
        Structured benchmark dictionary with environment metadata and kernel metrics.
    """
    if not has_cpp_core():
        raise BenchmarkError(
            "C++ vision core (football_cv._core) is not compiled. "
            "Please build the native extension before running micro-benchmarks."
        )

    logger.info(f"Starting micro-benchmarking suite (scale={scale})")
    env = EnvironmentCollector.collect()

    geo_iters = max(100, int(100_000 * scale))
    persp_iters = max(100, int(20_000 * scale))
    cam_iters = max(50, int(2_000 * scale))

    geo_results = benchmark_geometry(iterations=geo_iters)
    persp_results = benchmark_perspective(iterations=persp_iters)
    cam_results = benchmark_camera_motion(iterations=cam_iters)

    all_results = geo_results + persp_results + cam_results

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scale": scale,
        "environment": env,
        "summary": {
            "total_benchmarks": len(all_results),
            "mean_speedup": round(float(np.mean([r.speedup for r in all_results])), 2),
            "max_speedup": round(float(np.max([r.speedup for r in all_results])), 2),
            "min_speedup": round(float(np.min([r.speedup for r in all_results])), 2),
        },
        "benchmarks": [r.to_dict() for r in all_results],
    }


def format_micro_benchmark_table(report: dict[str, Any]) -> str:
    """
    Render micro-benchmark results into a formatted ASCII / Markdown table.
    """
    lines = [
        "=========================================================================================",
        "                     football-cv: Micro-Benchmark Performance Matrix                     ",
        "=========================================================================================",
        f"{'Kernel Operation':<34} | {'Python (μs)':<11} | {'C++ (μs)':<10} | {'Speedup':<8} | {'Category'}",
        "-----------------------------------------------------------------------------------------",
    ]

    for b in report.get("benchmarks", []):
        name = b["name"]
        py_lat = f"{b['python_latency_us']:.3f}"
        cpp_lat = f"{b['cpp_latency_us']:.3f}"
        sp = f"{b['speedup']:.2f}x"
        cat = b["category"]
        lines.append(f"{name:<34} | {py_lat:<11} | {cpp_lat:<10} | {sp:<8} | {cat}")

    summary = report.get("summary", {})
    lines.append(
        "-----------------------------------------------------------------------------------------"
    )
    lines.append(
        f"Mean Speedup: {summary.get('mean_speedup', 'N/A')}x  |  "
        f"Peak Speedup: {summary.get('max_speedup', 'N/A')}x  |  "
        f"Total Kernels: {summary.get('total_benchmarks', 0)}"
    )
    lines.append(
        "========================================================================================="
    )
    return "\n".join(lines)


def save_micro_benchmark_results(
    report: dict[str, Any], output_path: str | Path
) -> Path:
    """
    Persist benchmark report to JSON artifact.
    """
    path = Path(output_path)
    if path.is_dir() or path.suffix == "":
        path = path / "micro_results.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved micro-benchmark results to {path}")
    return path
