# Webcam Depth Detection — Research Roadmap

Single webcam, relative depth, CPU real-time, research/experimentation.

## 1. Can you get depth from one webcam?

**Yes.** A single 2D webcam frame contains enough cues (texture
gradient, perspective, occlusion, defocus) for a neural network to
estimate a depth map. This is called **monocular depth estimation
(MDE)**. Output is typically *relative* depth (ordinal: closer/farther),
not metric (meters), unless you use a metric model or calibrate.

There are three families of approaches:

| Family | Idea | Fits us? |
|---|---|---|
| Learning-based MDE | A CNN/ViT predicts depth per pixel from one RGB frame | Yes — primary path |
| Geometric / SfM | Recover depth from camera motion across frames (parallax) | Partial — assumes static scene |
| Depth from defocus | Use blur cues; needs known optics | No — webcams not suitable |

We will pursue learning-based MDE. SfM is a stretch goal for static
backgrounds.

## 2. Models to evaluate (CPU real-time, relative depth)

Recommended shortlist for a laptop CPU at ~15+ FPS:

| Model | Size | License | ONNX | Notes |
|---|---|---|---|---|
| MiDaS v2.1 small (256) | ~21M | MIT | Official | Lowest-friction baseline. ~15-30 FPS CPU. |
| MiDaS v3.1 dpt_swin2_tiny | ~40M | MIT | Official | Better quality, ~5-10 FPS CPU. |
| Depth Anything V2 - Small | ~25M | Apache 2.0 | Community | Current SOTA at small sizes. ~6-12 FPS ONNX, 12-20 FPS with OpenVINO INT8. |
| Depth Anything V2 - Base | ~98M | Apache 2.0 | Community | Quality reference, 1-3 FPS CPU. |

Skip on CPU but useful as quality references on GPU/offline:
ZoeDepth (metric), Metric3D V2, UniDepth V2, Marigold (diffusion),
Apple DepthPro, MoGe, Video Depth Anything.

**Starter recommendation:** MiDaS-small as baseline -> swap to
Depth Anything V2-Small via OpenVINO INT8 to compare quality vs FPS.

## 3. Runtime & speed-up techniques

- **ONNX Runtime** (`onnxruntime`) on CPU with `ORT_ENABLE_ALL`
  graph optimization, multi-threading via `intra_op_num_threads`.
- **OpenVINO** (Intel) gives the largest CPU win — INT8 PTQ via NNCF
  often 2-3x faster than fp32 ONNX. See `openvinotoolkit/openvino_notebooks`
  ("vision-monodepth", "depth-anything").
- **DirectML** (`onnxruntime-directml`) opportunistically uses iGPU.
- **Input resolution** — drop to 224 or 256 (from default 384/518) for
  big speedups; quality cost is moderate for relative depth.
- **Quantization** — INT8 with VNNI on modern CPUs is the single
  biggest lever. fp16 on CPU helps less.
- **Frame skipping** — run depth every Nth frame, interpolate between.

## 4. Inference pipeline (OpenCV + ONNX)

Standard webcam loop pattern:

1. `cv2.VideoCapture(0)` -> BGR frame.
2. BGR -> RGB, resize with aspect-preserving letterbox to model input
   (256/384/518), normalize with ImageNet mean/std, HWC->CHW, add batch.
3. `session.run(None, {input_name: tensor})`.
4. Resize depth back to frame size (`cv2.resize`, bilinear).
5. Robust normalize to 0-255 uint8 (percentile clip 1st/99th to kill
   outliers, *not* raw min-max).
6. Colormap with `cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)`.
7. Display side-by-side with input via `cv2.imshow`.

This belongs as a new module — proposed paths:

- `src/depth/base.py` — `BaseDepthEstimator` ABC.
- `src/depth/midas_onnx.py` — MiDaS-small ONNX wrapper.
- `src/depth/depth_anything_onnx.py` — Depth Anything V2 ONNX wrapper.
- `scripts/run_depth.py` — webcam demo, mirrors `run_webcam.py`.

## 5. Integrating depth with the existing pipeline

Existing flow: `Camera -> VideoStream -> Detector -> Analytics -> Display`.

Four placements to consider (research/effort tradeoff):

1. **Parallel branch + per-detection sampling (recommended).** Depth
   model runs alongside detector; we sample depth at each bbox.
   Lowest coupling, no retraining. Best research/effort ratio.
2. **4-channel RGBD detector.** Concatenate depth as 4th channel into
   YOLO. Requires retraining and good metric/calibrated depth — heavy.
3. **Depth as tracking feature.** Append median depth to ByteTrack /
   BoT-SORT cost matrix. Helps disambiguate ID switches when people
   cross at different distances. Low cost, moderate gain.
4. **Visualization-only.** Heatmap / pseudo-3D overlay. Zero risk, no
   algorithmic gain. Good for demos.

For research, do **(1)** first, then experiment with **(3)**.

### 5.1 Per-detection depth sampling — pseudocode

```
for each frame:
    rgb = capture()
    detections = yolo(rgb)               # N x (xyxy, cls, conf)
    depth_raw = depth_model(resize(rgb, model_in))
    depth = resize(depth_raw, rgb.shape[:2])
    depth_n = percentile_normalize(depth, 1, 99)
    for det in detections:
        roi = depth_n[shrink(det.bbox, 0.2)]   # avoid bg leak
        roi = reject_outliers(roi, mad=True)
        det.depth = median(roi)                # median > mean > center
```

Caveats:

- Detector and depth nets usually have **different input sizes** —
  always upsample depth to detector resolution.
- Outputs are **relative inverse depth** — normalize per frame, treat
  as ordinal.
- **Centre-pixel sampling fails** when bbox catches background.
  Shrink box 15-25% + masked median + MAD outlier rejection.
- SAM2 mask-conditioned sampling is strictly better but doubles cost.

## 6. Research topics — what to study

Group into the order you should learn them.

### 6.1 Foundations
- Camera intrinsics, pinhole model, focal length, FoV. Why MDE has
  a fundamental scale ambiguity.
- Disparity vs depth vs inverse depth. What MiDaS / Depth Anything
  actually output.
- ImageNet normalization, letterbox preprocessing, model input sizes.
- ONNX runtime providers (CPU, OpenVINO, DirectML, CUDA).

### 6.2 Core MDE methods
- **MiDaS** (Ranftl 2020 + DPT 2021) — relative depth, mixed-data
  training.
- **Depth Anything V1/V2** (Yang 2024) — large-scale unlabeled +
  DINOv2 backbone.
- **ZoeDepth** (Bhat 2023) — relative+metric hybrid heads.
- **Metric3D V2 / UniDepth** — camera-aware metric depth.
- **Marigold / Lotus / GenPercept** — diffusion-based depth (offline,
  reference quality).
- **Video Depth Anything** (2025) — temporal consistency.

### 6.3 Deployment & efficiency
- ONNX export from PyTorch (`torch.onnx.export`, `optimum-cli`).
- OpenVINO conversion + INT8 PTQ via NNCF.
- Quantization-aware training basics.
- Knowledge distillation (ViT teacher -> mobile student).

### 6.4 Integration with detection / tracking
- Per-detection sampling strategies (centre, median, masked, SAM).
- Tracking association with depth as additional cost term.
- Pseudo-3D bbox lifting from 2D + depth + intrinsics.
- Occlusion ordering from depth.

### 6.5 Calibration toward metric
- Single-anchor metric calibration (e.g., person ~1.7m).
- Linear fit `metric = a / (relative + b)` per session.
- Why this is brittle (depth drift, normalization range).

### 6.6 Open problems (worth a paper / blog)
- **Temporal consistency** — flicker, scale drift, EMA / one-Euro
  filters per track.
- **Edge artifacts** — hair, foliage, transparent surfaces.
- **Domain shift** — webcam exposure/AWB vs training data.
- **Dynamic scenes** — SfM/COLMAP poisoned by moving objects.
- **Camera-intrinsics generalization** — UniDepth / Metric3D V2
  approach.

## 7. Use cases this unlocks for the project

- **3D zone counting** — extend existing 2D polygon zones to depth
  prisms; count only objects in `[d_near, d_far]`.
- **Depth-aware anomaly** — alert when track depth drops below
  threshold (object too close) or depth-velocity spikes.
- **Occlusion-aware tracking** — front object's depth < rear's;
  freeze rear's appearance update during occlusion.
- **Pseudo-3D bbox visualization** — wireframe cuboid from frustum
  slice.
- **Distance overlays** in privacy / safety mode — "person at 1.4m".
- **Motion parallax fusion** — combine optical flow with MDE on
  static structures for scale refinement.

## 8. Suggested experiments (research log entries)

Each experiment: hypothesis, method, dataset/clips, metrics, result.

1. **Baseline MiDaS-small @ 256 on CPU** — measure FPS, qualitative
   depth on 5 indoor + 5 outdoor webcam clips.
2. **MiDaS vs Depth Anything V2-Small** — same clips, side-by-side
   FPS and edge sharpness comparison.
3. **OpenVINO INT8 quantization** — FPS and AbsRel delta vs fp32.
4. **Per-detection sampling strategies** — center vs median vs
   shrunk-box-MAD median; measure variance per static object.
5. **Tracking with depth feature** — IDF1 / ID switches on staged
   crossing scenarios with and without depth in BoT-SORT cost.
6. **Single-anchor metric calibration** — error in meters at 1m, 2m,
   3m, 5m using person-height anchor.
7. **Temporal smoothing** — raw vs EMA vs one-Euro on per-track depth
   sequence; measure jitter (std-dev under static object).

## 9. Evaluation methodology

- **Depth-only**: AbsRel, RMSE, delta<1.25 on NYU Depth V2 (indoor),
  KITTI (driving), DIODE (mixed). Median-scale relative outputs first.
- **Detection/tracking impact**: mAP / MOTA / IDF1 / ID-switch on
  MOT17 with-and-without depth feature; ablate weight.
- **Task-level**: zone-count accuracy on hand-labeled clips;
  precision/recall on staged "too-close" anomaly events.
- **Qualitative**: side-by-side heatmap, monotonic depth ordering on
  staged occlusions, per-track depth-over-time plots.

## 10. Pitfalls to avoid

- **Naive min-max normalization** — outliers crush dynamic range.
  Use 1st/99th percentile clipping.
- **Naive bbox-center sampling** — fails on thin/articulated objects;
  shrink box + masked median + outlier rejection.
- **Cross-frame absolute thresholds** on relative depth — the
  normalization range shifts as objects enter/leave. Use percentile
  bands or per-track relative deltas.
- **Latency stacking** — Depth Anything V2-Large will halve YOLO
  loop FPS. Use small/base, OpenVINO INT8, or run depth at half rate
  with interpolation.
- **Assuming static scene** — SfM/COLMAP-style refinement breaks on
  moving people; restrict to background regions only.
- **Trusting metric output without calibration** — even "metric"
  models drift; always sanity-check against a known-distance object.

## 11. Reading list / starter URLs

- github.com/isl-org/MiDaS
- github.com/DepthAnything/Depth-Anything-V2
- github.com/fabio-sim/Depth-Anything-ONNX
- github.com/openvinotoolkit/openvino_notebooks (search
  "depth-anything", "vision-monodepth")
- github.com/apple/ml-depth-pro
- github.com/prs-eth/Marigold
- github.com/lpiccinelli-eth/UniDepth
- github.com/microsoft/MoGe
- github.com/mikel-brostrom/boxmot (tracker hooks for depth feature)
- paperswithcode.com/task/monocular-depth-estimation (live leaderboard)
- huggingface.co/spaces (search "depth anything") for live demos

Papers:
- Ranftl et al., "Towards Robust Monocular Depth Estimation" (TPAMI 2020).
- Ranftl et al., "Vision Transformers for Dense Prediction" (ICCV 2021).
- Yang et al., "Depth Anything" (CVPR 2024).
- Yang et al., "Depth Anything V2" (NeurIPS 2024).
- Bhat et al., "ZoeDepth" (2023).
- Piccinelli et al., "UniDepth" (CVPR 2024).
- Ke et al., "Marigold" (CVPR 2024).

## 12. Suggested next milestones

| Phase | Deliverable |
|---|---|
| Depth-1 | `src/depth/midas_onnx.py` + `scripts/run_depth.py` baseline. |
| Depth-2 | Depth Anything V2-Small ONNX wrapper, OpenVINO INT8 build. |
| Depth-3 | Per-detection sampling integrated with YOLO output. |
| Depth-4 | Depth-aware zone counter (3D zones). |
| Depth-5 | Depth feature in tracker association. |
| Depth-6 | Single-anchor metric calibration utility. |
| Depth-7 | Temporal smoothing + per-track depth time-series. |

Log each experiment in `docs/research_log.md` with hypothesis,
method, results, conclusion.




