# Sample Assets & Input Video Guidelines

## 1. Provided Sample Footage

The repository references sample broadcast match footage located in `input_videos/`:

| File | Resolution | FPS | Total Frames | Duration | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`08fd33_4.mp4`** | $1920 \times 1080$ | 25.0 | 750 | 30.0s | `077985a55b61fe2ca413221c082711655b7ba736a3bbd8ff97b4dfc0e5b03c57` |

### Description
The clip depicts a continuous sequence of open-play football filmed from a elevated broadcast main-stand perspective:
* Panning and moderate zoom tracking player movement.
* Two distinct kit colors (Team 1 and Team 2) alongside the referee.
* Several contested ball possessions, passes, and ball transitions across the midfield.

---

## 2. Guidelines for User-Provided Match Video

When running the pipeline on custom footage, adhere to the following conditions for optimal tracking and analytics accuracy:

1. **Camera Angle:**
   - **Recommended:** Elevated tactical broadcast camera (wide angle showing 30–50% of the pitch).
   - **Avoid:** Ground-level camera, extreme close-ups, helmet/body cams, or severe vertical angles.
2. **Resolution & Framerate:**
   - Minimum $1280 \times 720$ at 25+ FPS.
   - 1080p ($1920 \times 1080$) at 25 or 30 FPS is standard.
3. **Camera Motion:**
   - Continuous pans and tilts are automatically compensated by the Lucas-Kanade optical flow estimator.
   - Avoid clips containing sudden camera cuts or instant replays, as scene transitions reset optical flow and ByteTrack track IDs.
4. **Pitch Calibration:**
   - If using a different pitch angle or section, update the 4-point homography coordinates in `configs/default.yaml` (`perspective.pixel_vertices`) to match known pitch landmarks (penalty box lines, touchlines, halfway line).
