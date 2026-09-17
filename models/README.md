# Model Weights Documentation & Provenance

This directory contains fine-tuned YOLOv8 model checkpoints specialized for football pitch object detection.

---

## 1. Checkpoint Inventory

| File | Size (Bytes) | SHA-256 Checksum | Purpose |
| :--- | :--- | :--- | :--- |
| **`best.pt`** | 18,503,665 (~17.6 MB) | `4182f74567a6acade9b39cd4317db5c5b44fca63c09f1332323fb749090c5cc0` | Primary production checkpoint (best validation mAP) |
| **`last.pt`** | 18,506,033 (~17.6 MB) | `0332e8a7833956a1a83a4791c061eb7f943bcdf246931e701389ec1bf9eaf9d4` | Final training epoch checkpoint |

---

## 2. Model Architecture & Class Mapping

* **Base Backbone:** Ultralytics YOLOv8 (Compact variant fine-tuned for football entities)
* **Classes (`m.names`):**
  - **`0: ball`** - The match football. Requires aggressive confidence thresholding ($conf \sim 0.10 - 0.15$) and temporal interpolation due to motion blur and small pixel footprint ($< 15 \times 15$ px).
  - **`1: goalkeeper`** - Dedicated class for goalkeepers. In downstream tracking, goalkeepers are merged with the player class for ByteTrack re-identification before jersey color classification.
  - **`2: player`** - Outfield football players from both teams.
  - **`3: referee`** - Match officials/referees. Tracked separately to prevent referee identities from being assigned team colors or ball possession.

---

## 3. Training & Provenance

* **Dataset:** Roboflow Football Players Detection Dataset.
* **Annotated Entities:** Bounding boxes around outfield players, goalkeepers, referees, and the ball across broadcast camera angles.
* **Input Resolution:** $640 \times 640$ pixels (during training and batch inference).
* **Inference Device:** Supports CUDA GPU acceleration or CPU execution (`device: auto`, `device: cpu`, `device: 0`).

---

## 4. Integrity & Verification

To verify that local model weights match the verified checksums:

```bash
# Windows (PowerShell):
Get-FileHash models/best.pt -Algorithm SHA256

# Linux / macOS:
sha256sum models/best.pt
```

Expected output for `models/best.pt`:
`4182f74567a6acade9b39cd4317db5c5b44fca63c09f1332323fb749090c5cc0`
