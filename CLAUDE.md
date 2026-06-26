# CLAUDE.md

Operating guide for AI agents (Claude Code) working in this repository. **Read this
first** — it is the root of the harness and encodes the invariants that CI enforces.

## What this is

A real-time **computer-vision R&D platform** (OpenCV + webcam / IP / RTSP / file
streams) in Python 3.11: object detection, tracking + re-identification, zones &
line-crossing, anomaly detection, pose/action, multi-camera fusion, depth, segmentation,
crowd counting, privacy, heatmaps, alerts, a Streamlit dashboard, and a workstation
ledger. See `README.md` for the feature/entrypoint table and `docs/system_architecture.md`
for the design.

## Golden rules (the invariants CI enforces)

1. **Lazy heavy imports / import-safety.** `src/` must import using *only* the slim deps
   (`numpy`, `opencv-python-headless`, `pyyaml`, `pillow`, `requests`, `scikit-learn`,
   `pytest`). Heavy/optional runtimes — `torch`, `ultralytics`, `supervision`,
   `transformers`, `onnxruntime`, `streamlit`, `torchreid` — **must be imported lazily
   inside the function/method that uses them**, never at module top level. This keeps the
   unit suite fast and CI ~2 GB lighter. Verify: `grep -nE "^(import|from) (torch|ultralytics|onnxruntime|transformers)" src/**/*.py` should return nothing.
2. **Test the pure core; mock the model.** Every new module ships unit tests for its
   *pure* logic (math, pre/post-processing, buffering, factory/config), with heavy models
   **mocked or injected**. Real inference is validated separately on a GPU/torch box — say
   so explicitly; never claim a model forward was validated here.
3. **Additive + config-gated.** New capabilities are off by default and selected via config
   or a factory (e.g. `build_detector`, `build_segmenter`, depth factory). Never change
   default behavior.
4. **Green per commit, torch-free.** `python -m pytest -q` must pass with only the slim
   deps installed before every commit.

## Dev commands

```bash
# Install the slim test deps (mirrors CI)
pip install pytest numpy pyyaml pillow requests opencv-python-headless scikit-learn

python -m pytest -q                         # full unit suite (torch-free)
python -m pytest tests/test_depth.py -q      # one file
python -m pytest -q -k reid                   # by keyword

# Optional heavy deps (NOT needed for tests/CI):
#   requirements.txt           runtime (ultralytics, torch, supervision, streamlit, …)
#   requirements-reid.txt      OSNet / torchreid re-ID embedder
#   requirements-phase2.txt    depth/SAM2/MNAD/P2PNet/CTR-GCN/XFeat/DINOv3 backends

python scripts/run_webcam.py [--model yolov8s.pt] [--track] [--reid]   # see README table
```

`tests/conftest.py` stubs `supervision`, `transformers`, `streamlit`, and `ultralytics`
so `src` imports cleanly without them; tests that assert loader behavior override the stub
via `monkeypatch`.

## Layout

| Path | Contents |
|------|----------|
| `src/models/` | `Detection` dataclass (`base.py`), detector **registry/factory** (`registry.py`), `yolo_wrapper`, `rtdetr_wrapper`, `open_vocab_detector` (OWLv2), `tiled_detector` (SAHI), `pose_detector`, `p2pnet_wrapper` (crowd), `dinov3_backbone` (embedding) |
| `src/analytics/` | `zone_counter`, `anomaly_detector` (+`SceneDescriptor`), `mnad_detector` (reconstruction anomaly), `heatmap`, `temporal`, `scene_understanding` |
| `src/tracking/` | `tracker` (EnhancedTracker), `reid` (pluggable embedder) |
| `src/multicam/` | `geometry` (homography/Hungarian), `manager` |
| `src/depth/`, `src/segmentation/`, `src/pose/` | Phase 2: depth (ONNX), SAM2 segmentation, CTR-GCN action |
| `src/deployment/` | `exporter` (ONNX / TensorRT / INT8) |
| `src/training/` | `active_learner`, `augmentation` |
| `src/utils/` | `drawing`, `features` (XFeat), `fps` |
| `src/workstation/` | workstation activity ledger |
| top-level `src/` | `alerts.py`, `privacy.py` (mask-aware), `dashboard.py` (Streamlit), `stream.py` |
| `scripts/` | `run_*.py` entrypoints + `benchmark*.py`, `export_model.py` |
| `tests/` | unit tests (`conftest.py` stubs heavy deps) |
| `configs/` | `default.yaml` |
| `docs/` | `system_architecture.md`, `research/`, **`ADLC.md`**, **`HARNESS.md`** |
| `.claude/` | harness: `workflows/`, `agents/`, `skills/`, `hooks/`, `settings.json` |

## Branch & PR conventions

- Develop on the **assigned feature branch**; never push to `main`/`master`.
- Conventional-commit messages (`feat(scope):`, `fix(scope):`, `docs:`…); keep CI green
  per commit; update `CHANGELOG.md` for user-facing changes.
- Open PRs as **draft**. Be frugal with PR comments.

## How we build here — the ADLC

`Research → Roadmap → Plan → Implement (lazy + tested) → Verify → Review → Ship`, run by a
small fleet of agents and deterministic workflows. The full lifecycle is **`docs/ADLC.md`**;
how the harness is wired to support it is **`docs/HARNESS.md`**. To author/run a multi-agent
workflow for a task, use the **`/workflow`** skill (`.claude/skills/workflow/`).
