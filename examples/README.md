# Sample Assets & Usage Examples

Guidelines, configuration templates, and execution recipes for running `football-cv` on sample broadcast footage.

---

## 1. Provided Sample Match Footage

The repository references sample broadcast match footage located in `input_videos/`:

| File | Resolution | FPS | Total Frames | Duration | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`08fd33_4.mp4`** | $1920 \times 1080$ | 25.0 | 750 | 30.0s | `077985a55b61fe2ca413221c082711655b7ba736a3bbd8ff97b4dfc0e5b03c57` |

### Footage Characteristics
The clip depicts open-play football from an elevated broadcast main-stand camera angle:
* Panning and moderate zoom tracking ball and player movement.
* Two distinct kit colors (Team 1 in white, Team 2 in dark red/maroon) alongside the referee (yellow).
* Multi-player ball possessions, passing transitions, and scrambles across the midfield.

---

## 2. Sample Configuration (`examples/sample_config.yaml`)

A fully-commented configuration template is provided at [`sample_config.yaml`](sample_config.yaml).

To run analysis using this configuration:
```bash
# Validate configuration parameters and asset availability
football-cv validate --config examples/sample_config.yaml

# Run video inference and analytics export
football-cv analyze --config examples/sample_config.yaml

# Generate tactical visualizations (heatmaps + passing networks)
football-cv report --config examples/sample_config.yaml --tracks outputs/analytics/events.json --output-dir outputs/report
```

---

## 3. Custom Match Video Guidelines

When applying `football-cv` to custom footage, follow these guidelines for optimal tracking and analytics accuracy:

1. **Camera Angle:**
   - **Recommended:** Elevated tactical broadcast camera (wide angle showing 30–50% of the pitch).
   - **Avoid:** Ground-level camera, extreme close-ups, handheld cameras, or severe vertical angles.
2. **Resolution & Framerate:**
   - Minimum $1280 \times 720$ at 25+ FPS.
   - $1920 \times 1080$ at 25 or 30 FPS is standard.
3. **Camera Motion:**
   - Smooth pans and tilts are automatically compensated by the Lucas-Kanade optical flow estimator.
   - Broadcast scene cuts are detected by the discontinuity filter ($> 25\text{ px/frame}$) and reset motion tracking to prevent homography explosion.
4. **Pitch Calibration:**
   - If using a different stadium or pitch angle, update the 4-point homography coordinates in your configuration YAML (`perspective.pixel_vertices`) to match known pitch landmarks (penalty box lines, touchlines, halfway line).
