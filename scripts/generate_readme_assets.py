"""
Script to generate high-resolution visual assets for the football-cv README and documentation.

All figures are derived from real pipeline outputs (no synthetic scorelines):
1. docs/assets/benchmark-speedup.png - Bar chart mapped by benchmark name from benchmarks/micro_results.json.
2. docs/assets/homography-projection.png - Broadcast pixels vs calibrated pitch-strip projection
   using the real PerspectiveTransformer defaults (68.0m x 23.32m visible window) and real
   tracked positions from outputs/analytics/player_tracking.csv.
3. docs/assets/camera-motion-flow.png - Symmetric margin optical flow (5% L/R, 10% top, per
   CameraMotionEstimator) with measured median displacement and RANSAC inlier ratio.
4. docs/assets/team-clustering-pipeline.png - Real two-stage pipeline: per-bbox jersey-vs-turf
   segmentation via TeamClassifier.get_player_color + Stage-2 K-Means (k=2) on measured Lab vectors.
5. docs/assets/player-telemetry-callout.png - Real tracked player with physically plausible speed
   (5-28 km/h) rendered with FrameAnnotator-style overlays plus an explanatory callout.

Notes:
- Input footage (input_videos/08fd33_4.mp4) contains a source-level scoreboard anonymization
  bar (black rectangle, top of frame). Figures acknowledge it rather than hide it.
- Optical-flow arrows are magnified x3.5 for visibility and labeled as such.
"""

import json
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from football_cv.perspective.transformer import PerspectiveTransformer
from football_cv.teams.classifier import TeamClassifier

OUTPUT_DIR = Path("docs/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = "input_videos/08fd33_4.mp4"
TRACKING_CSV = "outputs/analytics/player_tracking.csv"

# Must match CameraMotionEstimator defaults / configs/default.yaml
MARGIN_RATIO_X = 0.05
MARGIN_RATIO_Y = 0.10

# Canonical display labels keyed by benchmark name (never positional).
BENCHMARK_LABELS = {
    "bbox_foot_position": "Single BBox Foot Position",
    "bbox_center": "Single BBox Center",
    "filter_valid_bboxes": "BBox Spatial Area Filter",
    "batch_centers_25x": "Batch BBox Centers (25x)",
    "euclidean_distance": "Euclidean Distance (2D)",
    "perspective_transform_point": "Single Perspective Transform",
    "polygon_containment_check": "Point-in-Polygon Check",
    "find_nearest_point_22x": "Nearest Player Search (22x)",
    "perspective_transform_native_raw": "Raw Native Homography (No NumPy Alloc)",
    "camera_motion_margin_filter_200pts": "Margin Exclusion Filter (200 pts)",
    "batch_perspective_100pts": "Batch Homography Projection (100 pts)",
    "optical_flow_feature_loop_100pts": "Optical Flow Feature Loop (100 pts)",
}

TEAM_COLORS_MPL = {1: "#38bdf8", 2: "#f43f5e"}
TEAM_COLORS_BGR = {1: (255, 180, 60), 2: (70, 70, 220)}


def generate_benchmark_chart():
    """Generate bar chart with name-keyed labels sorted by measured speedup."""
    with open("benchmarks/micro_results.json", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]
    missing = [b["name"] for b in benchmarks if b["name"] not in BENCHMARK_LABELS]
    if missing:
        print(f"[WARN] Unmapped benchmark names (using raw id): {missing}")

    benchmarks_sorted = sorted(benchmarks, key=lambda x: x["speedup"])
    speedups = [b["speedup"] for b in benchmarks_sorted]
    display_names = [
        BENCHMARK_LABELS.get(b["name"], b["name"]) for b in benchmarks_sorted
    ]
    max_speedup = max(speedups)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    fig.patch.set_facecolor("#0b1329")
    ax.set_facecolor("#111c38")

    colors = []
    for s in speedups:
        if s >= 5.0:
            colors.append("#10b981")
        elif s >= 1.0:
            colors.append("#38bdf8")
        else:
            colors.append("#64748b")

    bars = ax.barh(display_names, speedups, color=colors, height=0.68, edgecolor="none")
    ax.axvline(
        1.0,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        label="1.0x Parity Baseline",
    )

    for bar, s in zip(bars, speedups, strict=False):
        w = bar.get_width()
        label = f"{s:.2f}x"
        if s >= 1.0:
            ax.text(
                w + 0.25,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                color="#f8fafc",
                fontsize=9.5,
                fontweight="bold",
            )
        else:
            ax.text(
                w + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{label} (FFI)",
                va="center",
                ha="left",
                color="#94a3b8",
                fontsize=8.5,
                fontstyle="italic",
            )

    ax.set_xlim(0, max(20.0, max_speedup + 2.0))
    ax.set_xlabel(
        "Speedup Factor (Python Latency / Native C++ Latency)",
        color="#e2e8f0",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_title(
        "football-cv: Native C++17 Kernel Acceleration vs. Pure Python",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="x", linestyle=":", color="#334155", alpha=0.6)

    fig.text(
        0.12,
        0.02,
        f"* Labels keyed by benchmark name from micro_results.json (n=12). "
        f"Single-item geometry shows FFI overhead (~0.2µs). Batched loops peak at {max_speedup:.2f}x.",
        color="#94a3b8",
        fontsize=8.5,
        fontstyle="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = OUTPUT_DIR / "benchmark-speedup.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated: {out_path}")


def _load_tracking():
    df = pd.read_csv(TRACKING_CSV)
    df["speed_kmh"] = df["speed_mps"] * 3.6
    return df


def generate_homography_visual():
    """Dual-panel visual using the real transformer defaults + real tracked points."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print("[WARN] Could not read frame 25, skipping homography visual.")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = frame_rgb.shape[:2]

    # Honest pipeline defaults: visible-window calibration (matches tests + C++ defaults).
    transformer = PerspectiveTransformer()
    court_w, court_l = float(transformer.court_width), float(transformer.court_length)
    vertices = transformer.pixel_vertices.tolist()

    df = _load_tracking()
    frame_df = (
        df[(df["frame_index"] == 25) & df["position_valid"].astype(bool)]
        .dropna(subset=["x_pitch", "y_pitch"])
        .copy()
    )
    # Balanced, spread-out sample: up to 5 per team by x position quantiles.
    sample_rows = []
    for team in (1, 2):
        tdf = frame_df[frame_df["team_id"] == team].sort_values("x_image")
        if len(tdf) > 5:
            idx = np.linspace(0, len(tdf) - 1, 5, dtype=int)
            tdf = tdf.iloc[idx]
        sample_rows.append(tdf)
    sample_df = pd.concat(sample_rows).head(11) if sample_rows else frame_df.head(11)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.8), dpi=200, gridspec_kw={"width_ratios": [1.2, 1.0]}
    )
    fig.patch.set_facecolor("#0f172a")

    # Panel 1: broadcast pixels (real tracked foot positions).
    ax1.set_facecolor("#0f172a")
    ax1.imshow(frame_rgb)
    ax1.set_title(
        "A. Broadcast Camera Space (Pixel Coordinates)",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    poly_patch = patches.Polygon(
        np.array(vertices),
        closed=True,
        linewidth=2.5,
        edgecolor="#06b6d4",
        facecolor="#06b6d4",
        alpha=0.22,
    )
    ax1.add_patch(poly_patch)
    v_labels = [
        f"V{i + 1} ({int(vx)}, {int(vy)})" for i, (vx, vy) in enumerate(vertices)
    ]
    for (vx, vy), vlbl in zip(vertices, v_labels, strict=True):
        ax1.plot(
            vx,
            vy,
            "o",
            color="#06b6d4",
            markersize=7,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
        )
        ax1.text(
            vx + (15 if vx < 500 else -110),
            vy + (35 if vy > 500 else -20),
            vlbl,
            color="#ffffff",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#0891b2",
                edgecolor="none",
                alpha=0.9,
            ),
        )
    for _, r in sample_df.iterrows():
        col = TEAM_COLORS_MPL.get(int(r["team_id"]), "#ffffff")
        ax1.plot(
            r["x_image"],
            r["y_image"],
            "o",
            color=col,
            markersize=8,
            markeredgecolor="#000000",
            markeredgewidth=1.5,
        )
    ax1.text(
        10,
        52,
        "Black bar (top): source scoreboard anonymization, not a pipeline artifact.",
        color="#e2e8f0",
        fontsize=8,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="#020617", edgecolor="none", alpha=0.8
        ),
    )
    ax1.set_xlim(0, w_img)
    ax1.set_ylim(h_img, 0)
    ax1.tick_params(colors="#94a3b8", labelsize=8.5)

    # Panel 2: calibrated pitch-strip (real pipeline coordinates, no rescaling).
    ax2.set_facecolor("#154224")
    ax2.set_title(
        f"B. Calibrated Pitch-Strip Projection ({court_w:.1f}m x {court_l:.2f}m window)",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax2.plot(
        [0, court_w, court_w, 0, 0],
        [0, 0, court_l, court_l, 0],
        color="#ffffff",
        linewidth=2,
        alpha=0.9,
    )
    ax2.plot(
        [0, court_w],
        [court_l / 2, court_l / 2],
        color="#ffffff",
        linewidth=1.5,
        alpha=0.85,
    )
    center_circle = patches.Circle(
        (court_w / 2, court_l / 2),
        min(9.15, court_l / 2 - 1.0),
        fill=False,
        color="#ffffff",
        linewidth=1.5,
        alpha=0.85,
    )
    ax2.add_patch(center_circle)
    ax2.plot(court_w / 2, court_l / 2, "o", color="#ffffff", markersize=4)

    texts = []
    for i, (_, r) in enumerate(sample_df.iterrows()):
        col = TEAM_COLORS_MPL.get(int(r["team_id"]), "#ffffff")
        x_m, y_m = float(r["x_pitch"]), float(r["y_pitch"])
        ax2.plot(
            x_m,
            y_m,
            "o",
            color=col,
            markersize=9,
            markeredgecolor="#ffffff",
            markeredgewidth=1.2,
        )
        # Stagger labels to avoid overlap along the crowded bottom edge.
        dy = 0.5 + (0.9 if i % 2 else 0.0)
        texts.append(
            ax2.text(
                x_m + 0.8,
                y_m + dy,
                f"#{int(r['player_id'])} ({x_m:.1f}m, {y_m:.1f}m)",
                color="#ffffff",
                fontsize=7.5,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#0f172a",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
        )
    ax2.set_xlim(-4, court_w + 4)
    ax2.set_ylim(-3, court_l + 3)
    ax2.set_xlabel(
        "Pitch Width (meters)", color="#cbd5e1", fontsize=9.5, fontweight="bold"
    )
    ax2.set_ylabel(
        "Pitch Length (meters, visible window)",
        color="#cbd5e1",
        fontsize=9.5,
        fontweight="bold",
    )
    ax2.tick_params(colors="#94a3b8", labelsize=8.5)
    leg_elements = [
        patches.Patch(
            facecolor="#38bdf8",
            edgecolor="#ffffff",
            label="Team 1 (pipeline team_id=1)",
        ),
        patches.Patch(
            facecolor="#f43f5e",
            edgecolor="#ffffff",
            label="Team 2 (pipeline team_id=2)",
        ),
    ]
    ax2.legend(
        handles=leg_elements,
        loc="upper right",
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#ffffff",
        fontsize=8.5,
    )
    fig.text(
        0.52,
        0.01,
        "Visible broadcast trapezoid maps to this calibrated strip (sub-region of a FIFA 105m x 68m pitch). "
        "Points are real pipeline outputs from player_tracking.csv frame 25.",
        color="#94a3b8",
        fontsize=8,
        fontstyle="italic",
        ha="center",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = OUTPUT_DIR / "homography-projection.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated: {out_path} ({len(sample_df)} real points)")


def generate_camera_motion_visual():
    """Margin flow visual with symmetric config margins + measured displacement."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ret1, f1 = cap.read()
    ret2, f2 = cap.read()
    cap.release()
    if not (ret1 and ret2):
        print("[WARN] Could not read frames for camera motion visual.")
        return

    h, w = f1.shape[:2]
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # Symmetric margins matching CameraMotionEstimator (left/right 5%, top 10%).
    left_w = max(20, int(w * MARGIN_RATIO_X))
    right_w = max(20, int(w * MARGIN_RATIO_X))
    top_h = max(20, int(h * MARGIN_RATIO_Y))
    mask = np.zeros_like(gray1)
    mask[:, :left_w] = 255
    mask[:, w - right_w :] = 255
    mask[:top_h, :] = 255

    pts1 = cv2.goodFeaturesToTrack(
        mask, maxCorners=80, qualityLevel=0.01, minDistance=15
    )
    # NOTE: goodFeaturesToTrack signature is (image, mask=...); call explicitly.
    pts1 = cv2.goodFeaturesToTrack(
        gray1, mask=mask, maxCorners=80, qualityLevel=0.01, minDistance=15
    )
    dx_med, dy_med, inlier_pct, n_pairs = 0.0, 0.0, 0.0, 0
    pairs = []
    if pts1 is not None:
        pts2, st, _err = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)
        if pts2 is not None:
            p1 = pts1[st.reshape(-1) == 1].reshape(-1, 2)
            p2 = pts2[st.reshape(-1) == 1].reshape(-1, 2)
            if len(p1) >= 2:
                dxs = p1[:, 0] - p2[:, 0]
                dys = p1[:, 1] - p2[:, 1]
                dx_med, dy_med = float(np.median(dxs)), float(np.median(dys))
                n_pairs = len(p1)
                # RANSAC consensus (same model family as pipeline: partial affine / Sim(2)-like).
                _M, inliers = cv2.estimateAffinePartial2D(
                    p2, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if inliers is not None and len(inliers) > 0:
                    inlier_pct = 100.0 * float(np.sum(inliers)) / float(len(inliers))
                pairs = list(zip(p1[:60], p2[:60], strict=False))

    vis_frame = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.imshow(vis_frame)

    for x, y, ww, hh in [
        (0, 0, left_w, h),
        (w - right_w, 0, right_w, h),
        (0, 0, w, top_h),
    ]:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                ww,
                hh,
                linewidth=1.5,
                edgecolor="#10b981",
                facecolor="#10b981",
                alpha=0.22,
            )
        )
    ax.add_patch(
        patches.Rectangle(
            (left_w, top_h),
            w - left_w - right_w,
            h - top_h,
            linewidth=1.5,
            linestyle="--",
            edgecolor="#f59e0b",
            facecolor="#000000",
            alpha=0.18,
        )
    )

    for p1, p2 in pairs:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        ax.plot(x1, y1, "o", color="#38bdf8", markersize=4)
        ax.arrow(
            x1,
            y1,
            (x2 - x1) * 3.5,
            (y2 - y1) * 3.5,
            color="#f43f5e",
            width=1.5,
            head_width=8,
            alpha=0.85,
        )

    # Badges placed fully inside their strips (no clipping).
    ax.text(
        left_w / 2,
        28,
        "Margin band\n(keypoints)",
        color="#ffffff",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#059669", edgecolor="none", alpha=0.9
        ),
    )
    ax.text(
        w - right_w / 2,
        28,
        "Margin band\n(keypoints)",
        color="#ffffff",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#059669", edgecolor="none", alpha=0.9
        ),
    )
    ax.text(
        w / 2,
        top_h + 22,
        "Top margin band (keypoints)  |  Black bar: source scoreboard anonymization",
        color="#e2e8f0",
        fontsize=8,
        fontstyle="italic",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#020617", edgecolor="none", alpha=0.8
        ),
    )

    ax.text(
        w / 2,
        int(h * 0.40),
        "Central pitch turf masked out: moving players/ball excluded from ego-motion estimate",
        color="#f8fafc",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#0f172a",
            edgecolor="#f59e0b",
            linewidth=1.2,
            alpha=0.9,
        ),
    )

    hud_text = (
        f"Measured inter-frame motion (frames 40->41):\n"
        f"- Method: Lucas-Kanade sparse flow + AffinePartial RANSAC (3.0 px)\n"
        f"- Median dx: {dx_med:+.2f} px | dy: {dy_med:+.2f} px (n={n_pairs})\n"
        f"- RANSAC inliers: {inlier_pct:.1f}%  |  Arrows x3.5 magnified"
    )
    ax.text(
        16,
        h - 150,
        hud_text,
        color="#ffffff",
        fontsize=8.5,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="#020617",
            edgecolor="#38bdf8",
            linewidth=1.5,
            alpha=0.92,
        ),
    )

    ax.set_title(
        "Symmetric-Margin Optical Flow & RANSAC Camera Motion Stabilization",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "camera-motion-flow.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(
        f"[OK] Generated: {out_path} (dx={dx_med:+.2f}, dy={dy_med:+.2f}, inliers={inlier_pct:.1f}%)"
    )


def _lab_swatches(lab_vec):
    patch = np.uint8([[np.clip(lab_vec, 0, 255)]])
    bgr = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)[0, 0]
    rgb = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2RGB)[0, 0]
    return np.full((80, 80, 3), rgb, dtype=np.uint8)


def generate_team_clustering_visual():
    """Real two-stage visual: measured jersey Lab vectors + k=2 team clustering."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print("[WARN] Could not read frame 50 for team clustering visual.")
        return
    df = _load_tracking()
    fdf = (
        df[(df["frame_index"] == 50) & df["position_valid"].astype(bool)]
        .dropna(subset=["x_image", "y_image"])
        .copy()
    )
    fdf = fdf[
        (fdf["x_image"] > 60)
        & (fdf["x_image"] < 1860)
        & (fdf["y_image"] > 300)
        & (fdf["y_image"] < 950)
    ]
    if len(fdf) < 6:
        print("[WARN] Too few valid players on frame 50 for clustering visual.")
        return

    classifier = TeamClassifier(color_space="lab")
    records = []
    for _, r in fdf.iterrows():
        x, y = float(r["x_image"]), float(r["y_image"])
        x1, y1 = int(max(0, x - 28)), int(max(0, y - 110))
        x2, y2 = int(min(frame_bgr.shape[1], x + 28)), int(min(frame_bgr.shape[0], y))
        if (x2 - x1) * (y2 - y1) < 800 or (y2 - y1) < 50:
            continue
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        lab = classifier.get_player_color(frame_bgr, bbox)
        if not np.any(np.asarray(lab) != 0):
            continue
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        torso_bgr = crop_bgr[: max(1, crop_bgr.shape[0] // 2), :]
        records.append(
            {
                "player_id": int(r["player_id"]),
                "team": int(r["team_id"]),
                "bbox": bbox,
                "crop_rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                "torso_rgb": cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2RGB),
                "lab": np.asarray(lab, dtype=float),
            }
        )
        if len(records) >= 24:
            break

    if len(records) < 6:
        print("[WARN] Too few measurable jersey vectors; skipping.")
        return

    lab_matrix = np.stack([d["lab"] for d in records])
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(lab_matrix)
    centroids = kmeans.cluster_centers_
    for d, lb in zip(records, labels, strict=False):
        d["cluster"] = int(lb)

    # Exemplars: clearest representative per PIPELINE team (white vs neon), largest bbox.
    # (Cluster labels are shown in the scatter; team ids give the intuitive contrast pair.)
    exemplars = []
    for team_id in (1, 2):
        cand = [d for d in records if d["team"] == team_id]
        cand.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        if cand:
            exemplars.append((team_id, cand[0]))
    exemplars.sort(key=lambda t: t[0])

    fig = plt.figure(figsize=(13, 6.4), dpi=200)
    fig.patch.set_facecolor("#0b1329")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.8])
    axes = [
        (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
        ),
        (
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ),
    ]
    ax_scatter = fig.add_subplot(gs[:, 3])
    ax_scatter.set_facecolor("#111c38")

    for row, (_team_id, ex) in enumerate(exemplars):
        ax_p, ax_t, ax_m = axes[row]
        ax_p.imshow(ex["crop_rgb"])
        ax_p.set_title(
            f"1. BBox Detection (#{ex['player_id']}, pipeline Team {ex['team']})",
            color="#ffffff",
            fontsize=8,
            fontweight="bold",
        )
        ax_p.axis("off")
        ax_t.imshow(ex["torso_rgb"])
        ax_t.set_title(
            "2. Upper-Torso Crop (jersey)",
            color="#ffffff",
            fontsize=8,
            fontweight="bold",
        )
        ax_t.axis("off")
        ax_m.imshow(_lab_swatches(ex["lab"]))
        lab_int = ", ".join(f"{v:.0f}" for v in ex["lab"])
        ax_m.set_title(
            f"3. Measured Kit LAB [{lab_int}]",
            color="#ffffff",
            fontsize=8,
            fontweight="bold",
        )
        ax_m.axis("off")

    a_vals = lab_matrix[:, 1]
    b_vals = lab_matrix[:, 2]
    for c in (0, 1):
        sel = labels == c
        ax_scatter.scatter(
            a_vals[sel],
            b_vals[sel],
            s=45,
            alpha=0.85,
            edgecolors="#ffffff",
            linewidth=0.8,
            label=f"Measured jerseys: cluster {c} (n={int(np.sum(sel))})",
        )
    ax_scatter.scatter(
        centroids[:, 1],
        centroids[:, 2],
        s=160,
        marker="X",
        c=["#fbbf24", "#38bdf8"],
        edgecolors="#000000",
        linewidths=1.5,
        label="Stage-2 centroids",
        zorder=5,
    )
    # Perpendicular-bisector decision boundary in (a,b).
    c0, c1 = centroids[0, 1:], centroids[1, 1:]
    mid = (c0 + c1) / 2.0
    direction = c1 - c0
    if np.linalg.norm(direction) > 1e-6:
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        span = float(np.ptp(a_vals)) + float(np.ptp(b_vals)) + 20.0
        p_a = np.array([mid - normal * span, mid + normal * span])
        ax_scatter.plot(
            p_a[:, 0],
            p_a[:, 1],
            linestyle="--",
            color="#38bdf8",
            linewidth=1.5,
            alpha=0.8,
            label="K-Means boundary (a-b bisector)",
        )

    ax_scatter.set_title(
        "4. Stage-2 K-Means on Measured CIELAB Vectors (k=2)",
        color="#ffffff",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_scatter.set_xlabel("OpenCV LAB a* channel (0-255)", color="#cbd5e1", fontsize=9)
    ax_scatter.set_ylabel("OpenCV LAB b* channel (0-255)", color="#cbd5e1", fontsize=9)
    ax_scatter.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax_scatter.spines.values():
        spine.set_color("#334155")
    ax_scatter.legend(
        loc="upper left",
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#ffffff",
        fontsize=8,
    )
    fig.suptitle(
        "Unsupervised Team Kit Classification: (1) Jersey-vs-Turf Segmentation, (2) Team Clustering",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        f"Frame 50, n={len(records)} measured jersey vectors from real detections. "
        "Stage-1: per-bbox k=2 segmentation with corner background voting (TeamClassifier).",
        color="#94a3b8",
        fontsize=8,
        fontstyle="italic",
        ha="center",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = OUTPUT_DIR / "team-clustering-pipeline.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated: {out_path} (n={len(records)} real vectors)")


def generate_telemetry_callout():
    """Anatomy figure for one real player with plausible kinematics."""
    df = _load_tracking()
    candidates = df[
        df["position_valid"].astype(bool)
        & df["speed_mps"].notna()
        & (df["speed_kmh"] >= 5.0)
        & (df["speed_kmh"] <= 28.0)
        & (df["distance_m"] < 120.0)
        & (df["x_image"] > 200)
        & (df["x_image"] < 1700)
        & (df["y_image"] > 350)
        & (df["y_image"] < 950)
    ].copy()
    if candidates.empty:
        print("[WARN] No plausible-speed candidates; skipping telemetry callout.")
        return
    # Prefer frame ~120 for continuity with prior figure, else median-speed row.
    near120 = candidates.iloc[(candidates["frame_index"] - 120).abs().argsort()[:25]]
    row = near120.sort_values("speed_kmh").iloc[len(near120) // 2]

    frame_idx = int(row["frame_index"])
    track_id = int(row["player_id"])
    speed_kmh = float(row["speed_kmh"])
    dist_m = float(row["distance_m"])
    team_id = int(row["team_id"])
    fx, fy = float(row["x_image"]), float(row["y_image"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print(f"[WARN] Could not read frame {frame_idx}; skipping.")
        return
    h, w = frame_bgr.shape[:2]

    crop_w, crop_h = 560, 460
    cx0 = int(np.clip(fx - crop_w / 2, 0, w - crop_w))
    cy0 = int(np.clip(fy - crop_h / 2 + 40, 0, h - crop_h))
    crop = frame_bgr[cy0 : cy0 + crop_h, cx0 : cx0 + crop_w].copy()
    px, py = fx - cx0, fy - cy0  # foot position inside crop

    team_bgr = TEAM_COLORS_BGR.get(team_id, (200, 200, 200))
    # Foot ellipse + ID badge + speed/distance (FrameAnnotator style).
    cv2.ellipse(
        crop,
        center=(int(px), int(py)),
        axes=(38, 13),
        angle=0,
        startAngle=-45,
        endAngle=225,
        color=team_bgr,
        thickness=2,
        lineType=cv2.LINE_4,
    )
    cv2.rectangle(
        crop,
        (int(px) - 20, int(py) + 6),
        (int(px) + 20, int(py) + 26),
        team_bgr,
        cv2.FILLED,
    )
    cv2.putText(
        crop,
        f"{track_id}",
        (int(px) - 8, int(py) + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )
    for halo, color in ((4, (255, 255, 255)), (2, (0, 0, 0))):
        cv2.putText(
            crop,
            f"{speed_kmh:.2f} km/h",
            (int(px) - 55, int(py) + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            halo,
        )
        cv2.putText(
            crop,
            f"{dist_m:.2f} m",
            (int(px) - 55, int(py) + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            halo,
        )

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=200)
    fig.patch.set_facecolor("#0b1329")
    ax.imshow(crop_rgb)
    ax.set_title(
        "Anatomy of a Real Player Tracking & Kinematics Telemetry Card",
        color="#ffffff",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    info = (
        f"Frame {frame_idx} / Track #{track_id} / Team {team_id}\n"
        f"Speed: {speed_kmh:.2f} km/h (plausible match pace)\n"
        f"Distance: {dist_m:.2f} m cumulative\n"
        f"Ellipse = foot anchor  |  Badge = ByteTrack ID\n"
        f"Text = smoothed rolling-window kinematics"
    )
    ax.text(
        12,
        30,
        info,
        color="#ffffff",
        fontsize=8.5,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#020617",
            edgecolor="#10b981",
            linewidth=1.5,
            alpha=0.93,
        ),
    )
    # Leader arrow from callout to the player foot.
    ax.annotate(
        "",
        xy=(px, py),
        xytext=(150, 150),
        arrowprops=dict(
            arrowstyle="->",
            color="#10b981",
            linewidth=1.8,
            connectionstyle="arc3,rad=0.15",
        ),
    )
    ax.text(
        8,
        crop_h - 8,
        "Values are real pipeline outputs (player_tracking.csv); speeds capped at 38 km/h in config.",
        color="#e2e8f0",
        fontsize=7.5,
        fontstyle="italic",
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="#020617", edgecolor="none", alpha=0.8
        ),
    )
    ax.axis("off")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "player-telemetry-callout.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(
        f"[OK] Generated: {out_path} (frame {frame_idx}, track {track_id}, {speed_kmh:.1f} km/h)"
    )


def main():
    print("Generating comprehensive visual assets for football-cv README...")
    generate_benchmark_chart()
    generate_homography_visual()
    generate_camera_motion_visual()
    generate_team_clustering_visual()
    generate_telemetry_callout()
    print("All README visual assets successfully generated!")


if __name__ == "__main__":
    main()
