# YOLOv8 Football Detector Training & Fine-Tuning Guide

This guide describes how to reproduce or fine-tune the YOLOv8 object detection model used in `football-cv`.

---

## 1. Dataset Preparation

The model is trained on a 4-class football detection schema:
- **`0: ball`** — The match football
- **`1: goalkeeper`** — Goalkeeper wearing a distinct kit
- **`2: player`** — Outfield players from both teams
- **`3: referee`** — Match officials

Prepare a standard YOLO dataset YAML configuration (e.g. `data/football.yaml`):

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
```

---

## 2. Fine-Tuning with Ultralytics YOLOv8

To train or fine-tune using the Ultralytics CLI:

```bash
# Fine-tune YOLOv8 medium on 1280x720 or 640x640 resolution
yolo detect train \
    data=data/football.yaml \
    model=yolov8m.pt \
    epochs=100 \
    imgsz=1280 \
    batch=16 \
    device=0 \
    patience=20 \
    save=True \
    project=runs/detect \
    name=football_yolov8m
```

### Python API Training Script:

```python
from ultralytics import YOLO

# Load base pretrained model
model = YOLO("yolov8m.pt")

# Train the model
results = model.train(
    data="data/football.yaml",
    epochs=100,
    imgsz=1280,
    batch=16,
    device=0,
    patience=20,
    save=True,
    project="runs/detect",
    name="football_yolov8m",
)
```

---

## 3. Model Export to ONNX

For high-throughput deployment with ONNX Runtime and the C++ acceleration core, export the trained checkpoint:

```bash
yolo export model=runs/detect/football_yolov8m/weights/best.pt format=onnx dynamic=True
```

Move the resulting checkpoints to the `models/` directory:
- `models/best.pt`
- `models/best.onnx`

Validate with pre-flight checks:
```bash
football-cv validate --config configs/default.yaml
```
