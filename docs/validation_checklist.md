# On-Hardware Validation Checklist

The unit suite (`python -m pytest -q`, 330 tests) validates the **pure cores** of every
module — math, pre/post-processing, factories, buffering, the depth-convention flip — with
heavy models **mocked** and **no torch/onnxruntime/GPU**. That is deliberately *not* the same
as validating that a model forward runs, produces accurate numbers, and hits real-time speed.

This checklist covers exactly that gap. Run it on real hardware with real footage before
relying on any item below. Each entry lists the **weights** (+ source/license), **deps**, the
**command/harness**, **what to measure**, and a **pass criterion**.

> Status legend: ✅ pure-core unit-tested in CI · ⚠️ model forward **not** run here (this list).
> Always re-check each model's **license** on its model card before any non-research use.

## 0. Environment setup

```bash
pip install -r requirements.txt            # ultralytics, torch, supervision, streamlit…
pip install -r requirements-reid.txt       # OSNet / torchreid (re-ID)
pip install -r requirements-phase2.txt     # onnxruntime, depth/SAM2/etc. backends
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Target devices to report against: a desktop GPU (e.g. RTX-class) **and** the intended edge box
(e.g. Jetson Orin / CPU). Record device + precision (FP32/FP16/INT8) with every number.

---

## A. CPU-only accuracy (no GPU — just a calibrated camera + a tape measure)

These items are pure-Python and already run; what's unverified is **real-world accuracy**.
Set up ground truth: place a person/object at measured distances (e.g. 1, 2, 3, 5, 8 m).

| # | Item | Command / harness | Measure | Pass criterion |
|---|------|-------------------|---------|----------------|
| 6 | **Ground-plane ranger** (`src/depth/ground_plane.py`) | Set `depth.ground_plane` (fx, fy, cx, cy, height_m, pitch_deg) from your camera calibration; feed a detection at each marked distance through `build_ground_ranger(...).detection_to_meters(det)` | abs / % range error vs tape at each distance | median error ≤ ~10–15% on flat ground within the calibrated range |
| 2 | **Scale calibration** (`DepthScaleCalibrator`) | Sample **raw** relative depth (`estimator.estimate`, not normalized) at ≥3 known distances → `fit(rel_samples, known_m)`; then check held-out distances | % error on held-out points | monotonic + error ≤ ~20% mid-range |
| 7 | **Time-to-collision** (`src/analytics/ttc.py`) | Walk toward the camera at a known steady speed; feed tracked boxes (and metric depth if available) to `TTCEstimator.update_batch` | predicted TTC vs (distance/speed) ground truth | TTC trend correct; within ~20–30% near contact; no false alarms when static/receding |
| — | **Relative-depth proximity** (`run_depth.py`, relative backend) | `python scripts/run_depth.py --depth-model <DAv2-S.onnx>` | does the alert fire only when objects are actually close? | qualitatively correct ordering (nearer ⇒ larger relative depth) |

---

## B. Metric depth #1 (GPU / ONNX Runtime)

| Field | Detail |
|-------|--------|
| **Weights** | Depth Anything V2 **Metric** — `depth-anything/Depth-Anything-V2-Metric-Hypersim-Small` (indoor, ~20 m) or `…-Metric-VKITTI-Small` (outdoor, ~80 m). Apache-2.0 (verify on card). Export to ONNX (e.g. the `fabio-sim/Depth-Anything-ONNX` exporter or `torch.onnx`). |
| **Deps** | `onnxruntime` (or `-gpu`); weights downloaded + exported to `.onnx`. |
| **Command** | `python scripts/run_depth.py --depth-backend depth_anything_metric --depth-model models/dav2_metric_s.onnx` (set `depth.proximity_threshold_m`). |
| **Measure** | (1) **Correctness of the convention flip**: nearer object ⇒ *smaller* meters, alert fires on `depth <= threshold_m`. (2) **Accuracy**: AbsRel / RMSE vs tape or a reference depth sensor. (3) **Speed**: end-to-end detect→depth→alert FPS at chosen `input_size`. |
| **Pass** | meters within ~AbsRel published for the model on your scene; alert direction correct; real-time (≥ ~10–15 FPS) at a usable `input_size` on the target box. |

---

## C. Phase-2 model modules (GPU / torch)

| # | Module | Weights / source (check license) | Command / harness | Measure | Pass |
|---|--------|----------------------------------|-------------------|---------|------|
| 2.2 | **SAM2 segmentation** (`src/segmentation/sam2_wrapper.py`) + mask-aware privacy | `sam2_b.pt` / `sam2_t.pt` via Ultralytics `SAM` (Apache-2.0) | `SAM2Segmenter(...).segment(frame, detections)` then `PrivacyFilter(...).apply` | masks align to objects; privacy blurs only masked pixels; FPS | masks sane; real-time enough for the use case |
| 2.3 | **MNAD anomaly** (`src/analytics/mnad_detector.py`) | none external — trains its own autoencoder on **your** normal footage (torch) | feed normal frames via `update()` until `is_fitted`, then `check()` on normal + staged-anomaly clips | ROC/AUC or TPR@FPR on held-out normal vs anomalous | separates anomalies from normal above chance with a usable threshold |
| 2.4 | **CTR-GCN action** (`src/pose/ctrgcn.py`) | a trained CTR-GCN checkpoint (e.g. NTU-RGB+D) exported to TorchScript + matching `labels` | feed a 17-kpt skeleton sequence from the pose stream to `classify()` | top-1 accuracy on a labeled clip set | matches the checkpoint's reported accuracy class on your actions |
| 2.5 | **P2PNet crowd** (`src/models/p2pnet_wrapper.py`) | P2PNet weights (ShanghaiTech A/B; original repo — research license) as TorchScript | `P2PNetCounter(model_path=…).count(frame)` | MAE/MSE vs hand-counted frames | count error within the dataset-reported MAE band |
| 2.6 | **XFeat** (`src/utils/features.py`) | `verlab/accelerated_features` via `torch.hub` (Apache-2.0) | `XFeatMatcher().match(frameA, frameB)` → `estimate_homography` | inlier ratio / reprojection error on an overlapping pair; FPS | stable matches, real-time on CPU/edge |
| 2.7 | **DINOv3 backbone** (`src/models/dinov3_backbone.py`) | `facebook/dinov3-vits16-*` via `transformers` (**check the DINOv3 license**; DINOv2-small is Apache-2.0 if you need permissive) | `DINOv3Backbone().embed(crop)`; compare same-ID vs diff-ID with `cosine_similarity` | same-ID cosine ≫ diff-ID; re-ID rank-1 if used for tracking | clear margin between same/different identities |

---

## D. Stereo subsystem #8 (calibrated stereo rig)

| Field | Detail |
|-------|--------|
| **Weights** | ESMStereo ONNX (official repo; MIT per the survey — verify). Export/obtain the `.onnx`. |
| **Hardware** | A calibrated stereo pair: known **baseline** (m) and **fx** (px), plus rectification maps (`cv2.stereoRectify` → `initUndistortRectifyMap`). |
| **Harness** | `rectify_stereo_pair(...)` → `m = build_stereo_matcher({"backend":"esmstereo","model_path":…})` → `disp = m.compute_disparity(L, R)` → `disparity_to_depth_map(disp, rig.fx, rig.baseline)` (ignore non-finite/invalid). |
| **Measure** | depth error vs tape at known distances; valid-pixel coverage; FPS at `input_size`. |
| **Pass** | Z error within ~a few % mid-range; runs at the target FPS (the survey cites ESMStereo ~91 FPS on AGX Orin — confirm on *your* box). |

---

## E. Temporal streaming depth #9 (GPU)

| Field | Detail |
|-------|--------|
| **Two paths** | (a) the shipped **`TemporalDepthEstimator`** smoother over any backend (`depth.streaming.enabled: true`) — validate it *reduces flicker*; (b) a true **oVDA** model as the inner estimator (**non-commercial weights**, GPU) for genuine temporal consistency. |
| **Harness** | `build_depth_estimator({... "streaming": {"enabled": true, "alpha": 0.5}})`; run on a video and compare per-pixel temporal variance with vs without smoothing. |
| **Measure** | temporal stability (std of depth at static pixels across frames) smoothing-on vs -off; added latency; (oVDA) accuracy vs single-frame. |
| **Pass** | measurable flicker reduction with acceptable lag; for oVDA, accuracy ≥ single-frame DAv2 with better temporal consistency. |

---

## Reporting

For each item record: **device + precision**, **input size**, **accuracy metric + value**,
**FPS**, pass/fail, and notes. File results under `docs/research/` or a `validation/` folder
and link them here so the ⚠️ items can graduate to ✅. Until then, treat all metric/accuracy/
FPS figures in this repo as *as-reported by sources*, not measured here.
