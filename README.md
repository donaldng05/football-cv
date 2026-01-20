# Football Computer Vision Analyzer

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)](https://opencv.org/)

A computer vision project that analyzes football match footage using YOLOv8 object detection to track players, estimate camera movement, calculate player speeds, and determine ball possession.

## ✨ Features
- **Multi-object tracking** using YOLOv8 and ByteTrack
- **Team classification** via K-means clustering on jersey colors
- **Camera movement compensation** using optical flow
- **Perspective transformation** to map pixel coordinates to real-world positions
- **Speed & distance calculation** for player performance metrics
- **Ball possession tracking** with player assignment

## 🛠️ Technical Stack
- **YOLOv8** (Ultralytics) for object detection
- **OpenCV** for image processing and optical flow
- **Supervision** library for tracking
- **scikit-learn** for K-means clustering
- **NumPy/Pandas** for numerical processing

## 📁 Project Structure

```text
football-cv/
├── camera_movement_estimator/   # Optical flow-based camera motion tracking
├── player_ball_assigner/        # Logic for ball possession detection
├── speed_and_distance_estimator/# Player performance metrics
├── team_assigner/               # K-means color clustering for teams
├── trackers/                    # YOLOv8 detection & ByteTrack
├── view_transformer/            # Perspective transformation
├── utils/                       # Helper functions for video & bbox ops
├── development_and_analysis/    # Jupyter notebooks for prototyping
├── main.py                      # Main pipeline
└── requirements.txt
\```

## 📦 Installation
\```bash
pip install ultralytics opencv-python supervision scikit-learn pandas numpy
\```

## 🚀 Usage
\```bash
python main.py
\```

## 🧠 What I Learned
- Implementing multi-object tracking pipelines
- Computer vision coordinate transformations
- Real-time video processing optimization
- Color-based clustering for team identification