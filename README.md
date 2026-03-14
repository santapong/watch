# OpenCV Object Detection Platform

R&D platform for real-time object detection, tracking, and computer vision research
using OpenCV and deep learning.

## Features

| Phase | Feature | Script |
|-------|---------|--------|
| 1 | YOLO Object Detection | `run_webcam.py` |
| 1 | Multi-source streaming (webcam, IP cam, RTSP, file) | `run_stream.py` |
| 2 | Zone counting & line crossing | `run_zone_counter.py` |
| 2 | Enhanced tracking with re-identification | `run_webcam.py --track --reid` |
| 2 | Open-vocabulary detection (OWLv2) | `run_open_vocab.py` |
| 3 | Anomaly detection from normal patterns | `run_anomaly.py` |
| 3 | Pose estimation & action recognition | `run_action.py` |
| 3 | Multi-camera grid view & fusion | `run_multicam.py` |
| 3 | Scene understanding & context | Built into main loop |
| 4 | Active learning sample collection | `run_active_learning.py` |
| 4 | Data augmentation pipeline | Library (`src/training/augmentation.py`) |
| 4 | Model export (ONNX, TensorRT) | `export_model.py` |
| 4 | Temporal event detection | Library (`src/analytics/temporal.py`) |

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run webcam detection
python scripts/run_webcam.py

# Run with specific model
python scripts/run_webcam.py --model yolov8s.pt

# Run with tracking enabled
python scripts/run_webcam.py --track
```

## Mobile Phone Camera

Connect your phone's camera as a wireless video source:

**Android (IP Webcam app):**
1. Install "IP Webcam" from Google Play Store
2. Open the app → tap "Start server"
3. Note the URL shown (e.g., `http://192.168.1.105:8080`)
4. Run:
```bash
python scripts/run_stream.py --url "http://192.168.1.105:8080/video"
```

**iOS:** Use "IPCamera - MJPEG Camera" app with the same approach.

**Requirements:** Phone and computer must be on the same WiFi network.

See [docs/system_architecture.md](docs/system_architecture.md) for detailed setup guides
(Android, iOS, RTSP cameras, troubleshooting).

## Scripts

```bash
# Basic detection
python scripts/run_webcam.py
python scripts/run_stream.py --url "http://PHONE_IP:8080/video"

# Zone counting (tracks objects in/out of defined areas)
python scripts/run_zone_counter.py

# Open-vocabulary detection (detect anything by text description)
python scripts/run_open_vocab.py --queries "fire extinguisher" "person with hat"

# Anomaly detection (learns normal, flags unusual)
python scripts/run_anomaly.py --learning-frames 300

# Pose estimation + action recognition
python scripts/run_action.py

# Multi-camera view
python scripts/run_multicam.py --sources 0 "http://PHONE_IP:8080/video"

# Active learning (collect uncertain samples for labeling)
python scripts/run_active_learning.py --session my_data

# Model export & benchmarking
python scripts/export_model.py --format onnx torchscript --benchmark
python scripts/benchmark.py --models yolov8n.pt yolov8n.onnx
```

## Controls

| Key   | Action                    |
|-------|---------------------------|
| `q`   | Quit                      |
| `s`   | Save screenshot           |
| `t`   | Toggle tracking on/off    |
| `c`   | Toggle confidence display |
| Space | Pause/resume              |

## Configuration

Edit `configs/default.yaml` to configure model, source, display, zones, anomaly
detection, temporal events, and more. See the
[configuration reference](docs/system_architecture.md#configuration-reference) for all options.

## Architecture

```
Camera → VideoStream → Detector → Analytics → Display
                         │
                    ┌────┴────┐
                    │ YOLO    │  Zone Counter  │  Anomaly  │  Action  │
                    │ OWLv2   │  Tracker       │  Temporal │  Scene   │
                    │ Pose    │  Multi-Camera  │  Learning │  Export  │
                    └─────────┘
```

See [docs/system_architecture.md](docs/system_architecture.md) for full architecture
diagrams and module documentation.

## Project Structure

```
src/
├── stream.py                     # Video source abstraction
├── models/
│   ├── base.py                   # Detection dataclass + BaseDetector ABC
│   ├── yolo_wrapper.py           # YOLO detector (Ultralytics)
│   ├── open_vocab_detector.py    # OWLv2 zero-shot detection
│   └── pose_detector.py          # Pose estimation + action classification
├── analytics/
│   ├── zone_counter.py           # Zone counting + line crossing
│   ├── anomaly_detector.py       # Scene anomaly detection
│   ├── scene_understanding.py    # Spatial relations + scene classification
│   └── temporal.py               # Time-based event detection
├── tracking/
│   └── tracker.py                # Enhanced tracking with re-ID
├── multicam/
│   └── manager.py                # Multi-camera grid + fusion
├── training/
│   ├── active_learner.py         # Uncertainty sampling for labeling
│   └── augmentation.py           # Data augmentation pipeline
├── deployment/
│   └── exporter.py               # Model export + benchmarking
└── utils/
    ├── drawing.py                # Visualization helpers
    └── fps.py                    # FPS counter

scripts/                          # Runnable scripts for each feature
configs/default.yaml              # Configuration file
docs/system_architecture.md       # Architecture + mobile camera guide
```

## Research Directions

See [idea.md](idea.md) for innovation ideas and research questions across 4 tiers,
from zone analytics to generative data augmentation.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot open video source` | Check camera index or URL. Verify same WiFi network for IP cameras |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Slow detection | Use smaller model (`yolov8n.pt`), lower resolution, or export to ONNX |
| High latency on phone stream | Use 5GHz WiFi, lower resolution in IP Webcam app |
| CUDA out of memory | Use smaller model or set `device: "cpu"` in config |
| OWLv2 download fails | Ensure internet connection for first-time HuggingFace model download |
