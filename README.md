# OpenCV Object Detection

R&D project for real-time object detection using OpenCV and deep learning.

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

# Run from phone camera (install IP Webcam app on Android)
python scripts/run_stream.py --url "http://PHONE_IP:8080/video"
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

Edit `configs/default.yaml` to change model, source, display settings, and more.

## Project Structure

```
src/models/base.py       - Detection dataclass + abstract BaseDetector
src/models/yolo_wrapper.py - YOLO model wrapper (Ultralytics)
src/stream.py            - Video source abstraction (webcam, IP cam, RTSP)
src/utils/drawing.py     - Visualization helpers
src/utils/fps.py         - FPS counter
scripts/run_webcam.py    - Main detection script
scripts/run_stream.py    - Network stream detection
configs/default.yaml     - Default configuration
```

See [idea.md](idea.md) for research directions and innovation ideas.
