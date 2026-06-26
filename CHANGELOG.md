# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **On-hardware validation checklist** (`docs/validation_checklist.md`): per unvalidated item
  (metric depth, ground-plane, TTC, calibration, SAM2, MNAD, CTR-GCN, P2PNet, XFeat, DINOv3,
  stereo, oVDA) — which weights/source/license, deps, the exact command/harness, what to
  measure (accuracy + FPS), and a pass criterion. README updated with the new depth/distance
  features and a Documentation section.
- **P2 distance scaffolds (roadmap #8–#9, lazy + config-gated):** a **real-time stereo
  subsystem** (`src/stereo/` — `BaseStereoMatcher`, ONNX `ESMStereo` with lazy onnxruntime,
  `StereoRig`, `build_stereo_matcher`) with pure, tested disparity→depth math
  (`Z = fx·baseline/disparity`, invalid disparity → +inf); and a **temporal streaming-depth
  smoother** (`src/depth/streaming.py` — `TemporalDepthEstimator`) that EMA-blends consecutive
  depth maps for steadier video depth (a lightweight stand-in for a true streaming model like
  oVDA, whose weights are non-commercial), wired as an opt-in wrapper in `build_depth_estimator`
  with `reset()` added to the depth base interface. Both need a stereo rig / GPU + weights, so
  they are **not inference-validated here**; their pure cores are unit-tested with the model mocked.
- **P1 distance/detection items (roadmap #5–#7):** YOLO26 made first-class (config example,
  edge-benchmark recipe via `scripts/benchmark_matrix.py`, and a regression test pinning the
  alias/loader — Ultralytics/YOLO weights are **AGPL-3.0**); a **single-view ground-plane
  ranger** (`src/depth/ground_plane.py` — `PinholeGroundRanger` + `GroundPlaneHomographyRanger`)
  giving true **meters on pure CPU** for fixed flat-ground cameras (no neural depth), reported
  through `is_too_close`; and a **time-to-collision** estimator (`src/analytics/ttc.py`)
  combining a unit-free bbox-looming cue with a metric range cue (range cue gated to metric
  units only — relative inverse-depth can't yield TTC). All additive, config-gated, pure
  numpy/cv2, unit-tested.
- **Metric depth + meters-denominated proximity** (roadmap item 1 from the distance/detection
  research): a `depth_anything_metric` backend in `build_depth_estimator` that returns **meters**,
  a `units` flag (`'relative'`|`'metric'`) on depth estimators, `Detection.depth_units`, and a
  convention-aware `is_too_close(depth, threshold, units)` — relative (larger = nearer) alerts on
  `>=`, metric (meters, smaller = nearer) alerts on `<=`. `run_depth.py` skips normalization on the
  metric path and alerts in meters via `depth.proximity_threshold_m`. **Additive + opt-in;** the
  default relative path's alerting/printing behavior is unchanged (it now also stamps
  `depth_units='relative'`). Pure logic unit-tested; real metric weights need a torch/ONNX box.
- **Approximate meters on the relative path + per-track range** (roadmap items 2–4):
  `DepthScaleCalibrator` (`src/depth/calibration.py`) fits `1/Z = a·d_rel + b` from a few
  known-distance references to convert relative depth to meters without swapping models;
  `prepare_depth_map(raw, units)` makes the metric "skip-normalization" rule a tested pure
  helper; a `--depth-input-size` knob on `run_depth.py` trades FPS/accuracy on edge; and
  `RangeKalman1D` / `RangeTracker` (`src/tracking/range_filter.py`) smooth per-track distance
  and estimate closing speed (range-rate), predicting through detection gaps. All pure numpy,
  additive, and unit-tested.
- **Research: real-time detection + distance estimation** (`docs/research/distance_detection_realtime_2026.md`):
  a web-grounded, adversarially-verified survey (10 domains, 10/10 fact-checked, 36 agents)
  of detectors, monocular/stereo/geometric depth, detection–depth fusion, collision/TTC,
  calibration, and edge deployment — plus a 9-item, repo-aware roadmap. Produced by the new
  `distance-detection-research` workflow (`.claude/workflows/`) per `docs/ADLC.md` (Research stage).
- **Workstation activity monitoring — Phase A** (`src/workstation/`): data model,
  SQLite-backed `ActivityStore`, and `ActivityLedger` with hysteresis debounce
  and auto-idle on observation silence. Taxonomy v1: polishing, filing,
  tool_change, idle, unknown. Includes `StaticAssignment` resolver for
  station→employee mapping (face/badge re-ID arrives in Phase B).
- `scripts/run_workstation.py` runner with `--demo` mode for offline pipeline
  validation; new `workstation:` section in `configs/default.yaml`.
- 40 new tests covering data classes, store round-trip + range queries, and
  ledger hysteresis / idle-timeout / transition accounting.
- **Pluggable detector backbones** (`src/models/registry.py`): a config-driven
  `build_detector` factory selecting YOLOv8/v10/v11/YOLO26 (one `YOLO` loader) or
  RT-DETR (`src/models/rtdetr_wrapper.py`) via `model.backend`. All run-scripts and
  the dashboard build detectors through the factory; `yolov8n.pt` stays the default.
- **INT8 quantization export** (`src/deployment/exporter.py`): `int8`/`data`
  calibration args on `ModelExporter.export` (ONNX/OpenVINO/TensorRT) with a
  half/int8 mutual-exclusion guard; `--int8`/`--data` flags on `export_model.py`.
- **Benchmark matrix** (`scripts/benchmark_matrix.py`): a model × precision × runtime
  latency/FPS/size (+optional mAP) sweep that probes-and-skips absent runtimes and
  writes `docs/research/data/benchmark_matrix.{json,md}` to pick a deployment default.
- **Continuous integration** (`.github/workflows/ci.yml`): runs the pytest suite on
  push/PR with a slim, ML-free dependency install.
- Import-safe detector wrappers (lazy `ultralytics` import) plus 19 new tests
  (registry factory, INT8 arg plumbing, result→`Detection` conversion).
- **Functional re-identification** (`src/tracking/reid.py` + `EnhancedTracker`): a
  pluggable `ReIDEmbedder` (zero-dep colour `HistogramEmbedder` default + optional deep
  `OSNetEmbedder` via torchreid, selected by `tracking.reid_backend: auto|histogram|osnet`).
  The tracker now maintains a lost-track pool and actually remaps a re-entering track's ID
  back to its original (previously the embeddings were computed but never matched).
  `EnhancedTracker` is now wired into `run_webcam.py --track` (re-ID + trajectory trails),
  not just the dashboard. Optional deps in `requirements-reid.txt` (CI stays torch-free).
- **Vectorized heatmap accumulation** (`src/analytics/heatmap.py`): the per-pixel Python
  loop is replaced by a precomputed circular-masked Gaussian kernel stamped via a clipped
  slice — identical output (verified to 1e-12), far faster per frame.
- **Multi-cue, debounced fall detection** (`ActionClassifier`): replaces the single
  `torso_angle < 30°` rule with a vote over torso-vs-horizontal angle (now direction-symmetric),
  lying-shaped bbox, and downward shoulder velocity, requiring N consecutive fall-like frames
  before reporting (graded confidence; tunable via `action.fall`). `run_action.py` now fires a
  critical, cooldown-limited fall alert through `AlertManager`. `pose_detector` imports
  `ultralytics` lazily.
- **SAHI-style tiled inference** (`src/models/tiled_detector.py`): `TiledDetector` wraps any
  `BaseDetector`, runs it over overlapping tiles, offsets boxes back to full-frame coords, and
  merges with class-aware NMS — recovering small/long-range objects. Enabled via `model.tiled`
  (off by default; the registry wraps the built detector when set). Tracking delegates to the
  full frame.
- **Geometry-grounded multi-camera identity** (`src/multicam/geometry.py` + `MultiCameraManager`):
  per-camera homographies project each detection's foot point into a shared bird's-eye view;
  `find_cross_camera_matches` now uses Hungarian assignment (scipy) on BEV distance (same class,
  within `max_match_distance`) and `assign_global_ids` gives matched people a shared ID across
  cameras (union-find). Falls back to the legacy class+confidence heuristic when uncalibrated.
  `run_multicam.py --config` loads per-camera homography and overlays shared IDs.
- **Monocular depth subsystem** (`src/depth/`, Phase 2): `BaseDepthEstimator` + ONNX
  estimators (Depth Anything V2 / MiDaS, lazy `onnxruntime`) selectable via
  `build_depth_estimator`, plus robust per-detection depth sampling (`sample_depth`:
  shrink + MAD + median) feeding a new `Detection.depth` field. `scripts/run_depth.py`
  overlays relative distance and fires proximity alerts via `AlertManager`. New `depth:`
  config block; optional deps in `requirements-phase2.txt` (the slim CI stays torch-free —
  the pure sampling/normalization/pre-post-process core is unit-tested with the ONNX session
  mocked).
- **Phase 2 model scaffolds** (lazy, config-gated, slim-CI-safe): promptable segmentation
  (`src/segmentation/` SAM 2) with a **mask-aware `PrivacyFilter`** that filters only masked
  pixels; a reconstruction (MNAD-style) anomaly backend conforming to the `AnomalyDetector`
  `update`/`fit`/`check` interface (`src/analytics/mnad_detector.py`); point-based crowd
  counting (`src/models/p2pnet_wrapper.py`); CTR-GCN skeleton action recognition
  (`src/pose/ctrgcn.py`); XFeat correspondence (`src/utils/features.py`); and a DINOv3
  embedding backbone (`src/models/dinov3_backbone.py`). Heavy runtimes are imported lazily;
  the pure cores (mask compositing, reconstruction threshold, point filtering, skeleton
  normalization, mutual-NN matching + homography, embedding normalize/cosine) are unit-tested
  with the models mocked. Real inference needs the optional `requirements-phase2.txt` deps +
  model weights, which can't be exercised in the slim CI.

## [0.5.0] - 2026-03-16

### Added
- **Heatmap generation** (`src/analytics/heatmap.py`): Density heatmaps from detection
  data with exponential decay, Gaussian blobs, colormap overlay, and snapshot export
- **Privacy mode** (`src/privacy.py`): GDPR/CCPA-compliant automatic blurring,
  pixelation, or blackout of detected persons/faces
- **Alert & notification system** (`src/alerts.py`): Configurable rule engine with
  cooldown, webhook (Slack/Discord), email (SMTP), and JSON log notifiers
- **Streamlit dashboard** (`src/dashboard.py`): Web-based monitoring UI scaffold with
  live feed, detection stats, heatmap overlay, and alert history panels
- Scripts: `run_heatmap.py`, `run_privacy.py`
- Configuration sections for heatmap, privacy, and alerts in `default.yaml`

### Improved
- **Test coverage**: Expanded from 6 to 114 unit tests covering detection, tracking,
  analytics (anomaly, scene, temporal), drawing utilities, heatmap, privacy, and alerts
- Added `tests/conftest.py` for clean dependency stubbing in test environment

## [0.4.0] - 2026-03-14

### Added
- **Active learning pipeline** (`src/training/active_learner.py`): Uncertainty sampling
  to identify low-confidence detections and export them for human labeling
- **Data augmentation** (`src/training/augmentation.py`): Geometric and photometric
  augmentations including cutout, mosaic, and YOLO-format dataset export
- **Edge deployment** (`src/deployment/exporter.py`): Export YOLO models to ONNX,
  TensorRT, OpenVINO, and other formats with benchmarking
- **Temporal event detection** (`src/analytics/temporal.py`): Detect time-based events
  like loitering, abandoned objects, crowd formation, and speed anomalies
- Scripts: `run_active_learning.py`, `export_model.py`, `benchmark.py`

## [0.3.0] - 2026-03-14

### Added
- **Anomaly detection** (`src/analytics/anomaly_detector.py`): Learn normal scene
  patterns using Isolation Forest, flag deviations automatically
- **Action recognition** (`src/models/pose_detector.py`): YOLOv8-pose skeleton
  extraction with rule-based action classification (standing, sitting, walking,
  running, falling, raising hand)
- **Multi-camera fusion** (`src/multicam/manager.py`): Manage multiple VideoStream
  instances with grid display and cross-camera detection matching
- **Scene understanding** (`src/analytics/scene_understanding.py`): Spatial relationship
  detection, scene type classification, and natural language scene descriptions
- Scripts: `run_anomaly.py`, `run_action.py`, `run_multicam.py`
- New drawing utilities: `draw_skeleton()`, `draw_action_label()`,
  `draw_anomaly_alert()`, `draw_scene_info()`, `draw_event_log()`

## [0.2.0] - 2026-03-14

### Added
- **Zone counting** (`src/analytics/zone_counter.py`): Polygon zone counting and
  line-crossing counters using the supervision library
- **Enhanced tracking** (`src/tracking/tracker.py`): Trajectory history, re-identification
  using color histogram embeddings, track statistics
- **Open-vocabulary detection** (`src/models/open_vocab_detector.py`): Zero-shot
  text-prompted object detection using HuggingFace OWLv2
- Scripts: `run_zone_counter.py`, `run_open_vocab.py`
- New drawing utilities: `draw_tracks()`, `draw_zones()`
- New dependencies: `scikit-learn`, `transformers`
- Extended configuration with zones, tracking, anomaly, multicam, temporal, and
  active learning sections

## [0.1.0] - 2026-03-14

### Added
- Initial project scaffold with YOLO object detection
- `VideoStream` class for webcam, IP camera (MJPEG), RTSP, and video file sources
- `YOLODetector` wrapper for Ultralytics YOLO models
- `BaseDetector` abstract class and `Detection` dataclass
- Real-time detection with FPS counter and model info display
- Keyboard controls: quit, screenshot, tracking toggle, confidence toggle, pause
- YAML-based configuration system
- Drawing utilities for bounding boxes, FPS, and model info
- Scripts: `run_webcam.py`, `run_stream.py`
- System architecture documentation with mobile camera setup guide
- Innovation ideas and research directions in `idea.md`
