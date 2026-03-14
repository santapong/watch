# System Architecture

## Overview

This document describes the architecture of the OpenCV Object Detection Platform — a modular,
extensible system for real-time computer vision research and applications.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                                │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Webcam   │  │ Mobile Phone │  │   RTSP   │  │  Video File  │   │
│  │ (USB/PCIe)│  │ (IP Webcam)  │  │  Camera  │  │  (mp4/avi)   │   │
│  └─────┬─────┘  └──────┬───────┘  └─────┬────┘  └──────┬───────┘   │
│        │               │                │               │           │
│        └───────────────┬┴────────────────┘───────────────┘           │
│                        │                                            │
│                        ▼                                            │
│              ┌─────────────────┐                                    │
│              │   VideoStream   │  Thread-safe frame grabber         │
│              │  (src/stream.py)│  Background thread for low latency │
│              └────────┬────────┘                                    │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DETECTION ENGINE                                │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  YOLODetector   │  │ OpenVocabDetector │  │   PoseDetector    │  │
│  │  (Ultralytics)  │  │ (OWLv2/HuggingFace)│ │  (YOLOv8-pose)   │  │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬──────────┘  │
│           │                    │                      │             │
│           └────────────────────┴──────────────────────┘             │
│                                │                                    │
│                    ┌───────────┴───────────┐                        │
│                    │  List[Detection]      │                        │
│                    │  (bbox, class, conf,  │                        │
│                    │   track_id, mask)     │                        │
│                    └───────────┬───────────┘                        │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ANALYTICS PIPELINE                                │
│                                                                     │
│  ┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────────┐   │
│  │   Zone     │ │   Anomaly    │ │  Action  │ │   Temporal     │   │
│  │  Counter   │ │  Detector    │ │ Recogn.  │ │   Events       │   │
│  └─────┬──────┘ └──────┬───────┘ └────┬─────┘ └───────┬────────┘   │
│        │               │              │                │            │
│  ┌─────┴──────┐ ┌──────┴───────┐ ┌────┴─────┐ ┌───────┴────────┐  │
│  │  Scene     │ │  Enhanced    │ │ Multi-   │ │   Active       │  │
│  │ Understand.│ │  Tracker     │ │ Camera   │ │   Learning     │  │
│  └────────────┘ └──────────────┘ └──────────┘ └────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT / DISPLAY                                │
│                                                                     │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────┐  │
│  │  CV2 Window│  │  Screenshots │  │  Alerts /  │  │   Export  │  │
│  │  (imshow)  │  │  (JPEG save) │  │  Logging   │  │  (ONNX/   │  │
│  │            │  │              │  │            │  │  TensorRT) │  │
│  └────────────┘  └──────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

```
src/
├── __init__.py
├── stream.py                    # VideoStream - unified input abstraction
│
├── models/                      # Detection engines
│   ├── base.py                  # Detection dataclass + BaseDetector ABC
│   ├── yolo_wrapper.py          # YOLO (Ultralytics) wrapper
│   ├── open_vocab_detector.py   # OWLv2 zero-shot text-prompted detection
│   └── pose_detector.py         # YOLOv8-pose skeleton extraction + action classification
│
├── analytics/                   # Analysis modules
│   ├── zone_counter.py          # Polygon zone counting + line crossing
│   ├── anomaly_detector.py      # IsolationForest on scene descriptors
│   ├── scene_understanding.py   # Spatial relationships + scene descriptions
│   └── temporal.py              # Time-based event detection (loitering, abandoned objects)
│
├── tracking/                    # Object tracking
│   └── tracker.py               # Enhanced tracker with re-ID + trajectory history
│
├── multicam/                    # Multi-camera support
│   └── manager.py               # Multi-stream manager with grid display
│
├── training/                    # Training utilities
│   ├── active_learner.py        # Uncertainty sampling + labeling queue
│   └── augmentation.py          # Data augmentation + synthetic generation
│
├── deployment/                  # Model export & benchmarking
│   └── exporter.py              # ONNX/TensorRT export + latency benchmarks
│
└── utils/                       # Shared utilities
    ├── drawing.py               # Visualization (boxes, skeletons, zones, tracks)
    └── fps.py                   # FPS counter
```

---

## Data Flow

```
Camera Frame (BGR numpy array)
       │
       ▼
  ┌─────────┐
  │ Detect   │──→ List[Detection(bbox, confidence, class_id, class_name, mask, track_id)]
  └─────────┘
       │
       ├──→ ZoneCounter.update(detections)     → zone counts, line crossings
       ├──→ EnhancedTracker.update(detections)  → trajectories, re-ID matches
       ├──→ AnomalyDetector.check(detections)   → anomaly_score, is_anomalous
       ├──→ ActionClassifier.classify(poses)     → action labels per person
       ├──→ SceneAnalyzer.analyze(detections)    → scene description, relationships
       ├──→ EventDetector.update(detections)     → triggered events list
       │
       ▼
  ┌──────────┐
  │ Annotate │──→ Frame with boxes, labels, zones, trajectories, alerts
  └──────────┘
       │
       ▼
  ┌──────────┐
  │ Display  │──→ cv2.imshow() / screenshot / log
  └──────────┘
```

---

## Mobile Phone Camera Integration Guide

### How It Works

Your mobile phone acts as a wireless IP camera. It streams video over your local WiFi
network using HTTP (MJPEG) or RTSP protocol. The detection system connects to this
stream just like any network camera.

```
┌──────────────┐     WiFi (same network)     ┌───────────────────┐
│  Mobile Phone │ ──────────────────────────→ │  Detection Server │
│  (IP Webcam)  │   HTTP MJPEG / RTSP        │  (this system)    │
│               │   e.g. :8080/video          │                   │
└──────────────┘                              └───────────────────┘
```

### Android Setup (IP Webcam App)

1. **Install**: Download "IP Webcam" from Google Play Store (by Pavel Khlebovich)
2. **Configure** (in-app settings):
   - Video preferences → Resolution: 640x480 or 1280x720
   - Video preferences → Quality: 50-70% (balance quality vs bandwidth)
   - Video preferences → Orientation: Landscape
3. **Start**: Tap "Start server" at the bottom
4. **Note the URL**: The app shows an address like `http://192.168.1.105:8080`
5. **Connect from this system**:
   ```bash
   python scripts/run_stream.py --url "http://192.168.1.105:8080/video"

   # Or with tracking enabled
   python scripts/run_stream.py --url "http://192.168.1.105:8080/video" --track

   # Or use the main script directly
   python scripts/run_webcam.py --source "http://192.168.1.105:8080/video"
   ```

### Android Setup (DroidCam)

1. **Install**: Download "DroidCam" from Google Play Store
2. **Install PC client**: Download DroidCam client from dev47apps.com
3. **Connect**: Enter phone's IP address in the DroidCam client
4. **Use**: DroidCam creates a virtual webcam — use webcam index directly:
   ```bash
   python scripts/run_webcam.py --source 1
   ```

### iOS Setup

1. **IPCamera (MJPEG)**: Install "IPCamera - MJPEG Camera" from App Store
   - Start the server, note the URL
   - Connect: `python scripts/run_stream.py --url "http://PHONE_IP:8080/video"`

2. **EpocCam**: Install "EpocCam" from App Store + desktop driver
   - Creates a virtual webcam on your computer
   - Use: `python scripts/run_webcam.py --source 1`

### RTSP Cameras (Professional/Security)

```bash
# Generic RTSP camera
python scripts/run_stream.py --url "rtsp://admin:password@192.168.1.100:554/stream1"

# Hikvision
python scripts/run_stream.py --url "rtsp://admin:pass@IP:554/Streaming/Channels/101"

# Dahua
python scripts/run_stream.py --url "rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0"
```

### Network Requirements

- **Same WiFi network**: Phone and computer must be on the same local network
- **Firewall**: Ensure the port (default 8080) is not blocked
- **Bandwidth**: For 720p at 30fps, expect ~2-5 Mbps bandwidth usage
- **Latency**: Typical latency is 100-300ms over WiFi. Use 5GHz WiFi for lower latency

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Cannot connect to phone stream | Verify both devices on same WiFi network |
| High latency / frame drops | Lower resolution to 640x480, use 5GHz WiFi |
| Stream disconnects frequently | Keep phone plugged in, disable battery optimization for the app |
| Black/frozen frame | Restart the camera app, check phone camera permissions |
| "Cannot open video source" error | Verify the URL format and port number |

### Using Config File Instead of CLI

Edit `configs/default.yaml`:
```yaml
source:
  type: "url"
  url: "http://192.168.1.105:8080/video"
```

Then simply run:
```bash
python scripts/run_webcam.py
```

---

## Configuration Reference

All settings are in `configs/default.yaml`:

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| model | name | Model file (yolov8n/s/m/l/x.pt) | yolov8n.pt |
| model | confidence | Min confidence threshold | 0.25 |
| model | iou_threshold | NMS IoU threshold | 0.45 |
| model | device | "" (auto), "cpu", "cuda:0", "mps" | "" |
| model | classes | Filter classes (null=all, [0]=person) | null |
| source | type | "webcam", "url", or "file" | webcam |
| source | webcam_index | Camera device index | 0 |
| source | url | IP camera / RTSP URL | "" |
| source | resolution | [width, height] or null | null |
| display | show_confidence | Show confidence % on boxes | true |
| display | show_track_id | Show track IDs | true |
| display | show_fps | Show FPS counter | true |
| tracking | enabled | Enable object tracking | false |
| zones | enabled | Enable zone counting | false |
| zones | polygons | List of polygon zone definitions | [] |
| anomaly | enabled | Enable anomaly detection | false |
| anomaly | learning_frames | Frames to learn normal | 500 |
| temporal | enabled | Enable temporal events | false |
| temporal | loiter_seconds | Loitering threshold | 30 |

---

## Deployment Options

### Local Desktop
- Direct webcam + cv2.imshow() display
- Best for development and testing
- Requires display (X11/Wayland)

### Edge Device (Raspberry Pi / Jetson)
- Export model to ONNX or TensorRT for faster inference
- Use `scripts/export_model.py` to convert models
- Jetson Nano: TensorRT provides 5-10x speedup

### Server / Headless
- Use `opencv-python-headless` (already in requirements)
- Process RTSP/IP camera streams without display
- Output to files, logs, or webhooks

---

## Phase Summary

| Phase | Feature | Status | Module |
|-------|---------|--------|--------|
| 1 | YOLO Detection + Streaming | Complete | src/models, src/stream |
| 2a | Zone Analytics & Counting | Complete | src/analytics/zone_counter |
| 2b | Enhanced Tracking + Re-ID | Complete | src/tracking/tracker |
| 2c | Open-Vocabulary Detection | Complete | src/models/open_vocab_detector |
| 3a | Anomaly Detection | Complete | src/analytics/anomaly_detector |
| 3b | Action Recognition | Complete | src/models/pose_detector |
| 3c | Multi-Camera Fusion | Complete | src/multicam/manager |
| 3d | Scene Understanding | Complete | src/analytics/scene_understanding |
| 4a | Active Learning | Complete | src/training/active_learner |
| 4b | Data Augmentation | Complete | src/training/augmentation |
| 4c | Edge Deployment | Complete | src/deployment/exporter |
| 4d | Temporal Detection | Complete | src/analytics/temporal |
