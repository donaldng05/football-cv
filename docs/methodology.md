# Methodology & Pipeline Architecture

## 1. Baseline Execution Environment

The baseline pipeline performance was audited and recorded under the following environment:

| Attribute | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 / Linux (Ubuntu 22.04 LTS compatible) |
| **Python Version** | Python 3.11.9 |
| **Video Processing** | OpenCV (`opencv-python` 4.12.x) |
| **Object Detection** | Ultralytics YOLOv8 (`ultralytics` 8.3.x) |
| **Multi-Object Tracking** | Supervision ByteTrack (`supervision` 0.26.x) |
| **Color Clustering** | Scikit-learn K-Means (`scikit-learn` 1.7.x) |
| **Reference Clip** | `input_videos/08fd33_4.mp4` (1920x1080, 25.0 fps, 750 frames) |

---

## 2. End-to-End Pipeline Stages

The analysis pipeline operates across 12 sequential stages, transforming raw video frames into annotated footage and structured analytics:

```text
[Input Video Frames]
        │
        ▼
 1. Object Detection (YOLOv8 on batches)
        │
        ▼
 2. Multi-Object Tracking (ByteTrack for players & referees)
        │
        ▼
 3. Camera Movement Estimation (Sparse optical flow on frame borders)
        │
        ▼
 4. Track Position Adjustment (Compensating for camera pan/tilt)
        │
        ▼
 5. Perspective Transformation (Homography to 2D pitch coordinates)
        │
        ▼
 6. Ball Position Interpolation (Pandas linear interpolation for missing ball detections)
        │
        ▼
 7. Speed & Distance Calculation (Metric displacement over rolling frame window)
        │
        ▼
 8. Team Assignment (K-Means jersey color clustering on cropped player bounding boxes)
        │
        ▼
 9. Ball Possession Assignment (Proximity matching between ball and player feet)
        │
        ▼
10. Temporal Hysteresis & Event Inference (State machine classifying passes, turnovers, recoveries)
        │
        ▼
11. Analytics Aggregation (Possession heatmaps & directed pass network graph generation)
        │
        ▼
12. Visualization & Export (Annotated video render + CSV/JSON structured event export)
```

---

## 3. Mathematical & Algorithmic Foundations

### A. Camera Motion Compensation
To differentiate camera movement from player physical movement across the pitch:
1. Feature points are identified along the frame margins (excluding the pitch interior where players run).
2. The Lucas-Kanade optical flow method estimates the displacement vector $(\Delta x_c, \Delta y_c)$ between consecutive frames.
3. Track pixel coordinates $(x, y)$ are adjusted:
   $$x_{\text{adj}} = x - \Delta x_c, \quad y_{\text{adj}} = y - \Delta y_c$$

### B. Perspective Homography
A planar perspective transform matrix $H$ is computed from four broadcast pixel vertices $P_{\text{img}}$ corresponding to known metric coordinates $P_{\text{pitch}}$ on the pitch:
$$p_{\text{pitch}} = H \cdot p_{\text{adj}}$$
Points outside the pitch polygon boundary are safely filtered to prevent erroneous off-pitch speed spikes.

### C. Team Classification via Color Space Clustering
1. The upper half of each player bounding box is cropped to isolate the jersey patch from shorts and pitch grass.
2. Two-stage K-Means clustering is executed:
   - *Stage 1:* Clusters jersey pixels from background pixels to extract the dominant jersey RGB vector.
   - *Stage 2:* Clusters all player jersey vectors into $k=2$ team clusters, assigning each player track to Team 1 or Team 2.

### D. Possession & Transition Inference
1. Euclidean distance $d(p_{\text{player}}, p_{\text{ball}})$ is measured at each frame.
2. If $\min(d) \le d_{\text{max}}$ (default 70 px), possession is tentatively assigned to that player.
3. A temporal hysteresis buffer ($N \ge 3$ frames) filters high-frequency assignment oscillations.
4. When possession transitions from Player $A$ to Player $B$:
   - If $\text{team}(A) == \text{team}(B)$: classified as a **`candidate_pass`**.
   - If $\text{team}(A) \ne \text{team}(B)$: classified as a **`turnover`**.
   - If transitioning from unassigned state: classified as a **`recovery`**.

### E. Possession Dominance Heatmaps
1. **Explainable Spatial Dominance**:
   Standard positional heatmaps measure raw player occupancy across the match, resulting in central congestion that fails to reflect tactical on-ball dominance. The possession heatmap strictly aggregates frames where `has_ball == True` and coordinates $(x_{\text{pitch}}, y_{\text{pitch}})$ are calibrated on the 2D tactical pitch.
2. **Duration Weighting ($\Delta t$)**:
   Each observation is weighted by the frame interval $\Delta t = 1 / \text{fps}$ (seconds) rather than raw frame detections. The resulting grid directly represents true on-ball control time in seconds:
   $$G[r, c] = \sum_{i \in \text{cell}(r, c)} \Delta t_i$$
3. **Continuous Density Smoothing**:
   A 2D Gaussian kernel ($G_{\sigma}$) is convolved over the discrete 2D histogram grid to yield smooth possession density contours while preserving total control duration.
4. **Artifacts & Data Delivery**:
   Figures are overlaid onto dimensionally accurate 2D tactical pitches (FIFA $105\text{ m} \times 68\text{ m}$) with dynamic alpha gradients (`outputs/report/heatmaps/team_{id}_possession.png`), alongside tabular grid density matrices (`outputs/report/data/possession_heatmap.csv`).

### F. Pass-Network Graph Inference
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
