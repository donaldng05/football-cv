# Methodology & Algorithmic Formulations

## 1. Baseline Execution Environment

The pipeline architecture and numerical algorithms are evaluated across standard benchmark environments:

| Attribute | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 / Linux (Ubuntu 22.04 LTS compatible) |
| **Python Runtime** | Python 3.11.9 (`football_cv` package) |
| **C++ Core Runtime** | C++17 (`football_cv_core` / `football_cv._core` extension via `pybind11`) |
| **Inference Engine** | Microsoft ONNX Runtime 1.20+ (C++ IntraOp thread pool) & PyTorch / Ultralytics |
| **Multi-Object Tracking** | Supervision ByteTrack (`supervision` 0.26+) |
| **Color Clustering** | Scikit-learn K-Means (`scikit-learn` in Lab color space) |
| **Reference Clip** | `input_videos/08fd33_4.mp4` (1920x1080, 25.0 fps, 750 frames) |

---

## 2. End-to-End Hybrid Processing Pipeline

The system operates across 12 sequential stages with dual-backend execution capability:

```text
[Broadcast Video Frames]
         │
         ▼
  1. Video Decoding (OpenCV VideoCapture / Streaming Generator)
         │
         ▼
  2. Object Detection (ONNX Runtime C++ / Ultralytics PyTorch with Class Confidences)
         │
         ▼
  3. Multi-Object Tracking (ByteTrack for players, referees, ball)
         │
         ▼
  4. Camera Motion Estimation (Sparse Optical Flow + 2D Sim(2) 4-DoF RANSAC)
         │
         ▼
  5. Dynamic Camera Compounding (H_eff = H_pitch @ H_cam)
         │
         ▼
  6. Planar Metric Homography (Projective transformation to 105m x 68m FIFA pitch)
         │
         ▼
  7. Ball Gap Interpolation (Temporal polynomial / linear gap filling)
         │
         ▼
  8. Kinematic Velocity & Distance (Smoothed metric displacement per delta_t)
         │
         ▼
  9. Team Kit Classification (Two-stage K-Means in Lab color space)
         │
         ▼
 10. Possession State Assignment (Proximity matching + temporal hysteresis)
         │
         ▼
 11. Tactical Analytics Aggregation (On-ball Gaussian heatmaps & pass network graphs)
         │
         ▼
 12. Visualization & Multimodal Export (Annotated video + 9 CSV/JSON schemas)
```

---

## 3. Mathematical & Algorithmic Foundations

### A. Camera Motion Compensation & 2D Sim(2) 4-DoF RANSAC

Broadcast match cameras undergo continuous panning, tilting, and optical zooming. To prevent camera motion from corrupting player pitch kinematics:
1. Feature keypoints are isolated exclusively in the outer broadcast margins (excluding the active pitch turf where players move).
2. Sparse Lucas-Kanade optical flow computes inter-frame feature displacements between frame $t-1$ and frame $t$.
3. When feature pairs exhibit complex motion (pan + zoom + slight rotation), a **minimal 2-point 2D Similarity / Affine Partial (Sim(2)) RANSAC solver** is executed in C++ (`football_cv::CameraMotionEstimator::estimate_affine_partial_ransac`).

The similarity transformation maps source features $(x, y)$ to target features $(x', y')$:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & -b & t_x \\ b & a & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
where:
* Scale factor $s = \sqrt{a^2 + b^2}$
* Rotation angle $\theta = \operatorname{atan2}(b, a)$
* Translation vector $\mathbf{t} = (t_x, t_y)$

**Minimal Closed-Form 2-Point Solution:**
Given two correspondence pairs $(p_1, q_1)$ and $(p_2, q_2)$:
$$\Delta x = p_1.x - p_2.x, \quad \Delta y = p_1.y - p_2.y, \quad D = \Delta x^2 + \Delta y^2$$
$$\Delta u = q_1.x - q_2.x, \quad \Delta v = q_1.y - q_2.y$$
$$a = \frac{\Delta x \Delta u + \Delta y \Delta v}{D}, \quad b = \frac{\Delta x \Delta v - \Delta y \Delta u}{D}$$
$$t_x = q_1.x - (a \, p_1.x - b \, p_1.y), \quad t_y = q_1.y - (b \, p_1.x + a \, p_1.y)$$

Hypothesis models are scored against all feature correspondences using squared reprojection error $r^2 \le \tau^2$ (default $\tau = 3.0\text{ px}$). The consensus model with maximum inlier support yields the robust inter-frame motion matrix.

---

### B. Perspective Homography & Dynamic Effective Compounding

A projective transformation maps image plane coordinates $(u, v)$ to metric pitch coordinates $(x, y)$:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{H}_{0 \to \text{pitch}} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad x = \frac{x'}{w'}, \quad y = \frac{y'}{w'}$$

In `football_cv::PerspectiveTransformer`, $\mathbf{H}_{0 \to \text{pitch}}$ is solved directly via an $8 \times 8$ linear system setup from 4 reference broadcast landmarks with gauge normalization $H_{22} = 1.0$.

When camera movement is present, the **effective homography matrix** compounds static calibration with camera motion $H_{t \to 0}$:
$$\mathbf{H}_{\text{eff}} = \mathbf{H}_{0 \to \text{pitch}} \cdot \mathbf{H}_{t \to 0}$$

Projected points are evaluated against horizon singularities ($w' \le 10^{-6}$) and boundary policies:
* **`Strict`**: Discards points falling outside the $105\text{m} \times 68\text{m}$ pitch rectangle (returns `nullopt`).
* **`Clip`**: Clamps coordinates to $[0, W_{\text{pitch}}] \times [0, L_{\text{pitch}}]$.
* **`Extrapolate`**: Permits unbounded coordinates beyond touchlines.

---

### C. Class-Specific Pre-NMS Confidence Filtering

Because sports broadcast footage contains severe class imbalance (the small ball occupies $< 0.05\%$ of frame pixels, while players occupy large vertical clusters), uniform confidence filtering induces high false-negative rates for the ball.

The native C++ ONNX detector decouples detection sensitivity:
$$\tau_{\text{conf}}(c) = \begin{cases} 0.12 & \text{if } c = \text{ball} \\ 0.25 & \text{if } c \in \{\text{player}, \text{goalkeeper}, \text{referee}\} \end{cases}$$

Candidate anchors are filtered against $\tau_{\text{conf}}(c)$ *prior* to Non-Maximum Suppression (NMS), preserving ball tracking continuity without flooding NMS queues with background clutter.

---

### D. Team Classification via Color Space Clustering

1. The upper half of each player bounding box is cropped to isolate the jersey patch from shorts, socks, and green turf.
2. Two-stage K-Means clustering is executed:
   - *Stage 1:* Clusters jersey pixels from background pixels to extract the dominant jersey RGB vector.
   - *Stage 2:* Clusters all player jersey vectors into $k=2$ team clusters in CIELAB color space, assigning each player track to Team 1 or Team 2.

---

### E. Possession & Transition Inference

1. Euclidean distance $d(p_{\text{player}}, p_{\text{ball}})$ is measured at each frame.
2. If $\min(d) \le d_{\text{max}}$ (default 70 px or 2.0 m in metric pitch space), possession is tentatively assigned to that player.
3. A temporal hysteresis buffer ($N \ge 3$ frames) filters high-frequency assignment oscillations.
4. When possession transitions from Player $A$ to Player $B$:
   - If $\text{team}(A) == \text{team}(B)$: classified as a **`candidate_pass`**.
   - If $\text{team}(A) \ne \text{team}(B)$: classified as a **`turnover`**.
   - If transitioning from unassigned state: classified as a **`recovery`**.

---

### F. Possession Dominance Heatmaps

1. **Explainable Spatial Dominance**:
   Standard positional heatmaps measure raw player occupancy across the match, resulting in central congestion that fails to reflect tactical on-ball dominance. The possession heatmap strictly aggregates frames where `has_ball == True` and coordinates $(x_{\text{pitch}}, y_{\text{pitch}})$ are calibrated on the 2D tactical pitch.
2. **Duration Weighting ($\Delta t$)**:
   Each observation is weighted by the frame interval $\Delta t = 1 / \text{fps}$ (seconds) rather than raw frame detections. The resulting grid directly represents true on-ball control time in seconds:
   $$G[r, c] = \sum_{i \in \text{cell}(r, c)} \Delta t_i$$
3. **Continuous Density Smoothing**:
   A 2D Gaussian kernel ($G_{\sigma}$) is convolved over the discrete 2D histogram grid to yield smooth possession density contours while preserving total control duration.
4. **Artifacts & Data Delivery**:
   Figures are overlaid onto dimensionally accurate 2D tactical pitches (FIFA $105\text{ m} \times 68\text{ m}$) with dynamic alpha gradients (`outputs/report/heatmaps/team_{id}_possession.png`), alongside tabular grid density matrices (`outputs/report/data/possession_heatmap.csv`).

---

### G. Pass-Network Graph Inference

1. **Graph Formulation & Node Identities**:
   Team ball circulation is modeled as a weighted directed graph $G = (V, E)$ overlaid on pitch dimensions:
   - **Nodes ($v \in V$):** Positioned at the empirical spatial centroid $(\bar{x}_{\text{pitch}}, \bar{y}_{\text{pitch}})$ of each player Track ID across possession and pass initiation events. Node radii scale with total on-ball involvement and possession duration. To maintain scientific integrity and avoid false identity assignment, nodes are labeled by computer-vision Track ID (e.g., `#10`) rather than unverified real-player roster names.
   - **Edges ($(u, v) \in E$):** Directed connections representing validated `candidate_pass` transitions from player $u$ to teammate $v$. Edge line thickness scales proportionally with pass volume, and arrow opacity maps to mean transition confidence.
2. **Noise Mitigation & Edge Pruning**:
   - *Temporal Windowing:* Transitions taking longer than `maximum_transition_frames` (default: 15 frames / 0.6s) without clear ball tracking are downgraded to `uncertain_transition` and excluded from pass-network edges.
   - *Edge Pruning:* A configurable `min_passes` threshold (default: 1 for short clips) eliminates low-frequency spurious handovers in dense match scrums.
3. **Exported Artifacts**:
   - Rendered tactical figures: `outputs/report/pass_networks/team_{id}_pass_network.png`
   - Tabular graph datasets: `outputs/report/data/pass_network_nodes.csv` and `pass_network_edges.csv`.
