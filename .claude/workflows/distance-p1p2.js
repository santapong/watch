/**
 * distance-p1p2
 * ---------------------------------------------------------------------------
 * Designs the P1+P2 distance/detection roadmap items (5-9) in parallel and
 * adversarially reviews each design, so the main loop can integrate them serially
 * (parallel file writes to one working tree would collide on configs/CHANGELOG, and
 * worktree outputs aren't merged back automatically — so this workflow is read-only:
 * Plan agents design, cv-reviewer red-teams; no agent writes the tree).
 *
 * Pipeline per item: design (Plan) -> review (cv-reviewer). No barrier.
 *
 * Run:  Workflow({ scriptPath: '.claude/workflows/distance-p1p2.js' })
 * Returns { designs: [{item, title, design, review}], stats }.
 */

export const meta = {
  name: 'distance-p1p2',
  description: 'Design + adversarially review the P1/P2 distance-detection roadmap items (YOLO26, ground-plane ranger, TTC, stereo, oVDA) for serial integration',
  whenToUse: 'Before implementing roadmap items 5-9 — produces a vetted, repo-grounded design per item.',
  phases: [
    { title: 'Design' },
    { title: 'Review' },
  ],
}

const PROJECT_CONTEXT = [
  'Repo: a real-time OpenCV + webcam CV platform (Python 3.11). Relevant seams:',
  '- Detectors: src/models/registry.py (build_detector / build_detector_from_config, _infer_family, already aliases "yolo26" -> YOLODetector), src/models/yolo_wrapper.py.',
  '- Detection dataclass: src/models/base.py (bbox x1y1x2y2, confidence, class_name, track_id, depth, depth_units).',
  '- Depth: src/depth/base.py (sample_depth, percentile_normalize, is_too_close(depth,threshold,units), prepare_depth_map, BaseDepthEstimator.units), src/depth/onnx_estimator.py (build_depth_estimator, OnnxDepthEstimator lazy onnxruntime, DepthAnythingV2 / DepthAnythingV2Metric), src/depth/calibration.py (DepthScaleCalibrator).',
  '- Geometry: src/multicam/geometry.py (homography + Hungarian helpers).',
  '- Tracking: src/tracking/range_filter.py (RangeKalman1D, RangeTracker -> smoothed range + range-rate), src/tracking/tracker.py.',
  '- Alerts: src/alerts.py (AlertManager, AlertRule). Entrypoints: scripts/run_*.py, scripts/benchmark*.py.',
  '- Tests: tests/ with conftest.py stubbing supervision/transformers/streamlit/ultralytics.',
].join('\n')

const GOLDEN_RULES = [
  'GOLDEN RULES (CI enforces): (1) heavy runtimes (torch/onnxruntime/ultralytics/transformers) imported LAZILY inside the function that uses them — never at module top. (2) Split a PURE core (math/pre-post/buffering/factory) and unit-test it with the model mocked/injected. (3) Additive + config-gated, off by default, never change existing defaults. (4) `python -m pytest -q` stays green with only slim deps (numpy, opencv-python-headless, pyyaml, pillow, requests, scikit-learn).',
].join(' ')

const ITEMS = [
  {
    key: 'yolo26',
    title: 'P1 #5 — YOLO26 detector config option + edge benchmark',
    brief: 'registry.py already aliases "yolo26" to YOLODetector and routes non-RT-DETR weights to it, so this is mostly additive: a config example (model.name: yolo26n.pt), a documented edge benchmark path via scripts/benchmark_matrix.py, a requirements.txt note (ultralytics version that ships YOLO26), and a unit test asserting the alias/loader with ultralytics mocked. NMS-free decoding => steadier detect->depth->alert latency; tighter small-object boxes => cleaner in-box depth sampling. Caveat: AGPL-3.0.',
  },
  {
    key: 'ground_plane',
    title: 'P1 #6 — Single-view ground-plane ranger (true meters on CPU)',
    brief: 'New PURE module (propose src/depth/ground_plane.py): for a fixed camera over flat ground, convert a detection foot-point (bbox bottom-centre) to metric distance Z. Support both parameterizations: (a) intrinsics (fx,fy,cx,cy) + camera height h + pitch; (b) a precomputed ground-plane homography (reuse src/multicam/geometry.py). Output meters (units="metric") so it plugs into is_too_close. Pure numpy/cv2 -> fully unit-testable with a synthetic camera giving known distances. Config-gated.',
  },
  {
    key: 'ttc',
    title: 'P1 #7 — Time-to-collision / looming estimator',
    brief: 'New PURE module (propose src/analytics/ttc.py): per-track time-to-contact. Two cues: (a) bbox looming — TTC = s / (ds/dt) from bbox width/area growth across frames; (b) range-based — TTC = Z / closing_speed using RangeTracker range + range-rate. Tracker-driven, keyed by track_id, robust to noise/zero-division. Pure numpy, unit-tested with synthetic sequences. Integration note: an AlertRule in run_depth or a new scripts/run_ttc.py.',
  },
  {
    key: 'stereo',
    title: 'P2 #8 — Real-time stereo subsystem (LAZY SCAFFOLD)',
    brief: 'New src/stereo/ package: BaseStereoMatcher + an ONNX stereo backend (e.g. ESMStereo) that lazy-imports onnxruntime, plus a factory build_stereo_matcher. PURE, testable core: disparity->metric depth Z = fx*baseline/disparity (guard disparity<=0), and a rectification helper note reusing multicam/geometry. Needs a stereo rig + weights -> NOT inference-validated here; ship as a lazy, config-gated scaffold with the model mocked, exactly like the Phase 2 modules.',
  },
  {
    key: 'ovda',
    title: 'P2 #9 — Temporally-consistent streaming depth, oVDA (LAZY SCAFFOLD)',
    brief: 'New stateful streaming-video-depth backend (propose a new class in src/depth/onnx_estimator.py or src/depth/streaming.py) implementing BaseDepthEstimator with cross-frame STATE + a reset(). PURE, testable core: the temporal smoothing (e.g. EMA / state-blend across consecutive depth maps) and reset behavior; the model forward stays lazy/mocked. Wire behind build_depth_estimator (off by default). Caveat: oVDA weights are NON-COMMERCIAL -> opt-in only.',
  },
]

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    item: { type: 'string' },
    summary: { type: 'string' },
    new_files: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          purpose: { type: 'string' },
          key_functions: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                signature: { type: 'string' },
                behavior: { type: 'string' },
              },
              required: ['name', 'signature', 'behavior'],
            },
          },
        },
        required: ['path', 'purpose', 'key_functions'],
      },
    },
    shared_edits: {
      type: 'array',
      items: {
        type: 'object',
        properties: { path: { type: 'string' }, change: { type: 'string' } },
        required: ['path', 'change'],
      },
    },
    config_keys: {
      type: 'array',
      items: {
        type: 'object',
        properties: { key: { type: 'string' }, default: { type: 'string' }, meaning: { type: 'string' } },
        required: ['key', 'meaning'],
      },
    },
    test_cases: { type: 'array', items: { type: 'string' } },
    lazy_imports: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    honest_caveats: { type: 'string' },
  },
  required: ['item', 'summary', 'new_files', 'test_cases', 'honest_caveats'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'should-fix', 'nit'] },
          issue: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'issue', 'fix'],
      },
    },
    verdict: { type: 'string', enum: ['approve', 'approve-with-nits', 'request-changes'] },
    notes: { type: 'string' },
  },
  required: ['findings', 'verdict'],
}

phase('Design')
const designs = await pipeline(
  ITEMS,
  // stage 1 — architect a concrete, repo-grounded design (read-only)
  it => agent(
    [
      `Design the implementation of: ${it.title}.`,
      it.brief,
      PROJECT_CONTEXT,
      GOLDEN_RULES,
      'Read the relevant files first to ground the design in the real code. Produce a concrete plan:',
      'new files with key function signatures + behavior, exact shared edits (config/__init__/CHANGELOG/requirements),',
      'config keys with defaults, the unit-test cases to write (what each asserts, model mocked), which heavy imports',
      'must stay lazy, risks, and an honest note on what CANNOT be validated in this torch-free env.',
      'DESIGN ONLY — do not modify any files.',
      `Set item to "${it.key}".`,
    ].join('\n'),
    { label: `design:${it.key}`, phase: 'Design', agentType: 'Plan', schema: DESIGN_SCHEMA },
  ),
  // stage 2 — adversarially review the design
  (design, it) => agent(
    [
      `Adversarially review this implementation design for "${it.title}". Try to find what will break or violate the repo invariants.`,
      'Check: correctness of the proposed math/logic; lazy imports (no top-level torch/onnxruntime/ultralytics); additive + config-gated (no changed defaults); pure core actually unit-testable with the model mocked; honest about what cannot be validated here; missing test cases or edge cases.',
      GOLDEN_RULES,
      `Design: ${JSON.stringify(design)}`,
    ].join('\n'),
    { label: `review:${it.key}`, phase: 'Review', agentType: 'cv-reviewer', schema: REVIEW_SCHEMA },
  ).then(review => ({ item: it.key, title: it.title, design, review })),
)

const out = designs.filter(Boolean)
log(`designed + reviewed ${out.length}/${ITEMS.length} items`)
return { designs: out, stats: { items: ITEMS.length, completed: out.length } }
