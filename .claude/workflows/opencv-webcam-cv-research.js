/**
 * opencv-webcam-cv-research
 * ---------------------------------------------------------------------------
 * A multi-agent (~58 opus agents) research team for the OpenCV + webcam
 * computer-vision platform in this repo. Surveys 24 CV domains, deep-reads the
 * top paper of each, adversarially verifies the headline citations/numbers,
 * synthesizes four thematic sections, and produces a lead-author report plus a
 * completeness critique. Every research agent grounds its claims in live web
 * search and is forbidden from inventing papers.
 *
 * Run:   Workflow({ name: 'opencv-webcam-cv-research' })
 *   or:  Workflow({ scriptPath: '.claude/workflows/opencv-webcam-cv-research.js' })
 *
 * Returns a report-ready object: { reportFrontMatter, sections[], references[],
 * verificationSummary, paperCards[], domainDigest[], critique, stats }.
 */

export const meta = {
  name: 'opencv-webcam-cv-research',
  description: 'Multi-agent (opus) research sweep on OpenCV + webcam computer vision: 24 domain surveys, deep-read paper cards, adversarial verification, thematic synthesis, cited report + critique',
  whenToUse: 'Deep, web-grounded research on the computer-vision state of the art relevant to a real-time OpenCV/webcam platform — produces a cited report and a project roadmap.',
  phases: [
    { title: 'Survey', detail: 'one opus researcher per CV domain (24) — web-grounded survey + real papers', model: 'opus' },
    { title: 'Deep-Dive', detail: 'deep-read each domain top paper into a structured card (24)', model: 'opus' },
    { title: 'Verify', detail: 'adversarial fact-check: papers real? headline numbers right? (4)', model: 'opus' },
    { title: 'Synthesize', detail: 'one cited markdown section per thematic cluster (4)', model: 'opus' },
    { title: 'Report', detail: 'lead-author front matter + roadmap, completeness critic (2)', model: 'opus' },
  ],
}

// ---------------------------------------------------------------------------
// Shared prompt fragments
// ---------------------------------------------------------------------------

const PROJECT_CONTEXT = [
  'Target project: an OpenCV + webcam real-time computer-vision platform (Python; ultralytics YOLO, opencv, torch, supervision, transformers, scikit-learn).',
  'It already implements: YOLO object detection, multi-source streaming (webcam/IP/RTSP/file), zone counting + line crossing, multi-object tracking with re-identification, open-vocabulary detection (OWLv2), scene anomaly detection, pose + action recognition, multi-camera grid + fusion, scene understanding, active learning, data augmentation, ONNX/TensorRT export + benchmarking, density heatmaps, privacy blurring, an alert system, a Streamlit dashboard, and a workstation activity ledger.',
  'The research must serve running advanced CV on consumer webcams and edge hardware in real time.',
].join(' ')

const WEB_RULES = [
  'You MUST ground every claim in live web search before answering — do not rely on memory; the current year is 2026.',
  'Use the WebSearch and WebFetch tools (and any mcp__Exa__* search tools). If a search tool is not loaded, call ToolSearch with query "select:WebSearch,WebFetch" (or a keyword) first.',
  'Prefer primary sources: arxiv.org, openaccess.thecvf.com, paperswithcode.com, official GitHub repos, and project pages.',
  'NEVER invent a paper, author, arXiv id, URL, or benchmark number. If you cannot verify a field, leave it blank. Accuracy beats completeness.',
].join(' ')

// ---------------------------------------------------------------------------
// Structured-output schemas
// ---------------------------------------------------------------------------

const PAPER_FIELDS = {
  title: { type: 'string' },
  authors: { type: 'string', description: 'e.g. "Liu et al."' },
  venue: { type: 'string', description: 'e.g. "CVPR 2024", "arXiv", "NeurIPS 2023"' },
  year: { type: 'integer' },
  arxivId: { type: 'string', description: 'e.g. "2304.07193"; blank if unknown' },
  url: { type: 'string' },
}

const SURVEY_SCHEMA = {
  type: 'object',
  properties: {
    domain: { type: 'string' },
    overview: { type: 'string', description: '180-260 word grounded overview of the state of the art' },
    keyTechniques: { type: 'array', items: { type: 'string' } },
    sotaMethods: { type: 'array', items: { type: 'string' }, description: 'Named SOTA methods/models, prefer 2023-2026, with approx accuracy/speed when known' },
    candidatePapers: {
      type: 'array',
      description: '3 to 5 REAL papers actually found via web search',
      items: {
        type: 'object',
        properties: Object.assign({}, PAPER_FIELDS, {
          whyImportant: { type: 'string' },
          keyClaim: { type: 'string', description: 'Headline quantitative result, with a number' },
        }),
        required: ['title', 'year', 'whyImportant'],
      },
    },
    relevanceToProject: { type: 'string', description: 'How this maps to the platform; name the concrete module/script' },
    openQuestions: { type: 'array', items: { type: 'string' } },
  },
  required: ['domain', 'overview', 'keyTechniques', 'sotaMethods', 'candidatePapers', 'relevanceToProject'],
}

const PAPERCARD_SCHEMA = {
  type: 'object',
  properties: {
    domain: { type: 'string' },
    paper: { type: 'object', properties: PAPER_FIELDS, required: ['title'] },
    problem: { type: 'string' },
    method: { type: 'string', description: 'How it works, 80-150 words' },
    headlineResults: { type: 'string', description: 'Concrete numbers: mAP / FPS / latency / params / memory' },
    datasets: { type: 'array', items: { type: 'string' } },
    limitations: { type: 'string' },
    realtimeWebcamApplicability: { type: 'string', description: 'Can it reach interactive FPS on CPU/consumer GPU? model size? what would it take on a webcam stream?' },
    integrationIdea: { type: 'string', description: 'Concrete change to this repo, naming the module/script' },
    claimsToVerify: { type: 'array', items: { type: 'string' }, description: '2-4 specific factual/numeric claims to independently confirm' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['domain', 'paper', 'method', 'headlineResults', 'realtimeWebcamApplicability', 'integrationIdea'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item: { type: 'string', description: 'Echo the [Pn] tag and the paper title' },
          tag: { type: 'string', description: 'The Pn id from the input, e.g. "P3"' },
          verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
          evidence: { type: 'string', description: 'What you found, with the source' },
          correction: { type: 'string', description: 'If refuted/uncertain, the correct fact; else blank' },
        },
        required: ['item', 'verdict', 'evidence'],
      },
    },
  },
  required: ['results'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'string' }, description: 'Important sub-topics/methods missed WITHIN the covered domains' },
    missingTopics: { type: 'array', items: { type: 'string' }, description: 'Entire areas not covered that a webcam-CV platform should consider' },
    weakestFindings: { type: 'array', items: { type: 'string' }, description: 'Claims that look thin or under-verified' },
    suggestedNextSteps: { type: 'array', items: { type: 'string' } },
    overallAssessment: { type: 'string' },
  },
  required: ['gaps', 'suggestedNextSteps', 'overallAssessment'],
}

// ---------------------------------------------------------------------------
// The 24 research domains (4 thematic clusters x 6)
// ---------------------------------------------------------------------------

const DOMAINS = [
  // Cluster A — Detection, Tracking & Identity
  { key: 'realtime-detection', cluster: 'A', title: 'Real-time object detection', module: 'src/models/yolo_wrapper.py, scripts/run_webcam.py',
    seeds: 'YOLOv8, YOLOv9, YOLOv10 (NMS-free), YOLO11, YOLOv12 (attention-centric), RT-DETR, RT-DETRv2, RTMDet, D-FINE, end-to-end real-time detection, COCO mAP vs latency tradeoff' },
  { key: 'open-vocab', cluster: 'A', title: 'Open-vocabulary / zero-shot detection', module: 'src/models/open_vocab_detector.py, scripts/run_open_vocab.py',
    seeds: 'OWL-ViT, OWLv2, GroundingDINO, Grounding DINO 1.5, YOLO-World, YOLOE, T-Rex2, open-vocabulary detection, zero-shot, LVIS, real-time text-prompted detection' },
  { key: 'mot', cluster: 'A', title: 'Multi-object tracking', module: 'src/tracking/tracker.py',
    seeds: 'ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, Hybrid-SORT, MOTRv2, MOT17, MOT20, DanceTrack, data association, Kalman filter' },
  { key: 'reid', cluster: 'A', title: 'Person re-identification', module: 'src/tracking/tracker.py (re-ID features)',
    seeds: 'OSNet, TransReID, CLIP-ReID, lightweight real-time re-identification, Market-1501, MSMT17, occluded ReID, cross-domain generalizable ReID' },
  { key: 'multicam', cluster: 'A', title: 'Multi-camera / cross-camera tracking & fusion', module: 'src/multicam/manager.py, scripts/run_multicam.py',
    seeds: 'multi-target multi-camera tracking MTMC, cross-camera ReID, camera topology, epipolar geometry, homography, WILDTRACK, AI City Challenge, 3D position from multiple views' },
  { key: 'small-object', cluster: 'A', title: 'Small / long-range object detection', module: 'src/models/yolo_wrapper.py (SAHI tiling)',
    seeds: 'small object detection, tiny object, SAHI slicing aided hyper inference, VisDrone, TinyPerson, high-resolution detection, P2 head, long-range from webcam' },

  // Cluster B — Spatial & Scene Understanding
  { key: 'segmentation', cluster: 'B', title: 'Promptable segmentation', module: 'src/models/ (new SAM2 segmenter)',
    seeds: 'Segment Anything SAM, SAM 2, MobileSAM, FastSAM, EdgeSAM, EfficientSAM, promptable segmentation, real-time SAM, video object segmentation, text-prompted segmentation' },
  { key: 'pose', cluster: 'B', title: 'Pose estimation', module: 'src/models/pose_detector.py, scripts/run_action.py',
    seeds: 'RTMPose, RTMO, ViTPose, YOLOv8-pose, YOLO11-pose, MediaPipe BlazePose, MoveNet, COCO keypoints, real-time 2D pose, whole-body pose, 3D pose from mono' },
  { key: 'depth', cluster: 'B', title: 'Monocular depth estimation', module: 'new src/models depth estimator + docs/webcam_depth_research.md',
    seeds: 'MiDaS, DPT, Depth Anything, Depth Anything V2, ZoeDepth, Metric3D, Metric3Dv2, UniDepth, monocular metric depth, real-time depth from a single webcam' },
  { key: 'scene-graph', cluster: 'B', title: 'Scene-graph generation', module: 'src/analytics/scene_understanding.py',
    seeds: 'scene graph generation, visual relationship detection, Neural Motifs, panoptic scene graph PSG, RelTR, open-vocabulary scene graph, Visual Genome' },
  { key: 'vlm', cluster: 'B', title: 'Vision-language models for scene description', module: 'src/analytics/scene_understanding.py (LLM scene description)',
    seeds: 'CLIP, BLIP-2, LLaVA, LLaVA-1.6, Qwen2-VL, Qwen2.5-VL, InternVL, MiniCPM-V, image captioning, visual question answering, on-device VLM' },
  { key: 'calibration', cluster: 'B', title: 'Camera calibration & geometry', module: 'src/multicam/manager.py + new src/utils calibration',
    seeds: 'camera calibration, Zhang method, intrinsics extrinsics, lens distortion, OpenCV calibrateCamera, homography, self-calibration, deep single-image calibration, fisheye' },

  // Cluster C — Video Intelligence & Anomaly
  { key: 'action-recognition', cluster: 'C', title: 'Skeleton-based action recognition', module: 'src/models/pose_detector.py + src/analytics/temporal.py, scripts/run_action.py',
    seeds: 'ST-GCN, 2s-AGCN, CTR-GCN, PoseConv3D / PoseC3D, skeleton action recognition, online action detection, NTU RGB+D, LSTM/transformer on pose sequences' },
  { key: 'video-anomaly', cluster: 'C', title: 'Video anomaly detection', module: 'src/analytics/anomaly_detector.py, scripts/run_anomaly.py',
    seeds: 'video anomaly detection, MNAD memory autoencoder, future frame prediction, Jigsaw, diffusion-based anomaly, UCF-Crime, ShanghaiTech, weakly-supervised MIL, unsupervised VAD' },
  { key: 'temporal', cluster: 'C', title: 'Temporal video understanding', module: 'src/analytics/temporal.py',
    seeds: 'VideoMAE, VideoMAE V2, TimeSformer, video transformer, online action detection, streaming, temporal action localization, long-term video memory, stateful detection' },
  { key: 'crowd-counting', cluster: 'C', title: 'Crowd counting & density estimation', module: 'src/analytics/heatmap.py, scripts/run_heatmap.py',
    seeds: 'crowd counting, density estimation, CSRNet, P2PNet, MCNN, DM-Count, point supervision, ShanghaiTech, occupancy estimation, congested scenes' },
  { key: 'classical-opencv', cluster: 'C', title: 'Classical OpenCV pipelines', module: 'src/utils + src/stream.py',
    seeds: 'optical flow Farneback Lucas-Kanade RAFT, background subtraction MOG2 KNN, ORB SIFT feature matching, OpenCV DNN module, contour analysis, Kalman, classical-vs-deep tradeoffs' },
  { key: 'fall-safety', cluster: 'C', title: 'Fall detection & workplace safety', module: 'src/models/pose_detector.py + src/alerts.py',
    seeds: 'fall detection, elderly monitoring, worker safety PPE detection, hard hat detection, construction safety compliance, vision-based fall datasets, healthcare activity monitoring' },

  // Cluster D — Models, Data & Deployment
  { key: 'edge-deploy', cluster: 'D', title: 'Edge deployment & model optimization', module: 'src/deployment/exporter.py, scripts/export_model.py, scripts/benchmark.py',
    seeds: 'ONNX Runtime, TensorRT, NCNN, OpenVINO, INT8 quantization (PTQ/QAT), structured pruning, knowledge distillation, Jetson, accuracy-latency-power tradeoff' },
  { key: 'active-learning', cluster: 'D', title: 'Active learning for detection', module: 'src/training/active_learner.py, scripts/run_active_learning.py',
    seeds: 'active learning object detection, uncertainty sampling, core-set, learning loss, BADGE, deep active learning, query strategy, labeling efficiency 5-10x' },
  { key: 'gen-aug', cluster: 'D', title: 'Generative / synthetic data augmentation', module: 'src/training/augmentation.py',
    seeds: 'diffusion synthetic data, Stable Diffusion / SDXL augmentation, copy-paste augmentation, DatasetDM, DA-Fusion, controllable generation, rare class, long-tail, sim2real' },
  { key: 'low-light', cluster: 'D', title: 'Low-light enhancement & image quality', module: 'src/stream.py preprocessing + src/utils',
    seeds: 'low-light image enhancement, Retinexformer, Zero-DCE, SCI, real-time denoising, motion deblur, lightweight super-resolution, HDR, exposure correction, webcam image quality' },
  { key: 'privacy', cluster: 'D', title: 'Privacy-preserving vision', module: 'src/privacy.py, scripts/run_privacy.py',
    seeds: 'privacy-preserving computer vision, face anonymization, DeepPrivacy, federated learning vision, differential privacy, on-device analytics, GDPR/CCPA, person de-identification' },
  { key: 'foundation-models', cluster: 'D', title: 'Vision foundation models / self-supervised', module: 'src/models/base.py (backbone/embeddings)',
    seeds: 'DINOv2, DINOv3, self-supervised vision, masked autoencoder MAE, vision foundation model, frozen backbone, linear probing, EVA, RADIO, general-purpose features' },
]

const CLUSTERS = [
  { key: 'A', title: 'A. Detection, Tracking & Identity' },
  { key: 'B', title: 'B. Spatial & Scene Understanding' },
  { key: 'C', title: 'C. Video Intelligence & Anomaly' },
  { key: 'D', title: 'D. Models, Data & Deployment' },
]

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

function surveyPrompt(d) {
  return [
    'You are a senior computer-vision research scientist writing a grounded survey of ONE domain.',
    'DOMAIN: ' + d.title + '.',
    PROJECT_CONTEXT,
    'Seed methods/keywords to investigate (not exhaustive — also find the very latest): ' + d.seeds + '.',
    WEB_RULES,
    'Do 4-8 targeted searches, then deliver: (1) a ~200-word grounded overview of the state of the art; (2) keyTechniques; (3) sotaMethods (named, prefer 2023-2026, with approximate accuracy/speed); (4) 3-5 candidatePapers you ACTUALLY found, each with accurate metadata (title, authors, venue, year, arxivId, url), whyImportant, and a keyClaim containing a number; (5) relevanceToProject naming the concrete repo module/script (' + d.module + '); (6) 2-3 openQuestions.',
    'Return ONLY the structured object.',
  ].join('\n\n')
}

function deepDivePrompt(d, survey) {
  const papers = ((survey && survey.candidatePapers) || []).map(function (p, i) {
    return (i + 1) + '. ' + p.title + ' (' + (p.venue || '') + ' ' + (p.year || '') + ') ' + (p.arxivId ? 'arXiv:' + p.arxivId : '') + (p.url ? ' ' + p.url : '')
  }).join('\n')
  return [
    'You are deep-reading the single most important paper for the domain: ' + d.title + '.',
    PROJECT_CONTEXT,
    'Candidate papers from the survey (pick the MOST important/impactful for real-time webcam/edge CV; you may pick a better one you find yourself):',
    papers || '(none provided — find the seminal or current-SOTA paper yourself)',
    'Seed keywords: ' + d.seeds + '.',
    WEB_RULES,
    'Fetch the chosen paper page (arXiv abstract / CVF / paperswithcode) and extract a structured paper card: problem; method (how it works, 80-150 words); headlineResults WITH NUMBERS (mAP/FPS/latency/params/memory); datasets; limitations; realtimeWebcamApplicability (concretely: interactive FPS on CPU or consumer GPU? model size? what it would take to run on a live webcam stream?); integrationIdea (a concrete change to this repo, naming the module/script ' + d.module + '); claimsToVerify (2-4 specific factual/numeric claims a fact-checker should confirm); confidence.',
    'Return ONLY the structured object.',
  ].join('\n\n')
}

function verifyPrompt(items) {
  return [
    'You are an ADVERSARIAL fact-checker for a computer-vision research report. Be skeptical by default.',
    'For each item below (a headline paper + its key claim), independently verify via web search:',
    '- Does the paper exist with the stated title/authors/venue/year? If the citation looks fabricated or metadata is wrong, mark "refuted" and give the correction.',
    '- Is the key numeric claim (mAP/FPS/latency/etc.) accurate per the primary source? If it materially disagrees, mark "refuted" with the correct figure. If you cannot confirm either way, mark "uncertain".',
    WEB_RULES,
    'Echo each item\'s Pn tag in the "tag" field so results can be matched back. Items:',
    items,
    'Return ONLY the structured object with exactly one result per item.',
  ].join('\n\n')
}

function synthPrompt(cluster, domainsJson, correctionsText) {
  return [
    'Write a rigorous, well-structured markdown SECTION for a computer-vision research report aimed at the OpenCV + webcam platform described below.',
    PROJECT_CONTEXT,
    'SECTION TITLE (use exactly): "## ' + cluster.title + '".',
    'Base the section ONLY on the grounded findings JSON below (per-domain surveys + deep-read paper cards). Respect the verification corrections: do NOT restate any refuted claim; hedge anything marked uncertain.',
    'For EACH sub-domain in this cluster include: a "### <sub-domain>" subsection with a short narrative of the state of the art; the most important methods/papers with inline citations like (FirstAuthor et al., Venue Year); a markdown comparison table when comparing methods on speed/accuracy/size; and a bold "**Use in this project:**" line tying to a concrete module/script. End the whole section with a "### Cluster insights" list of 2-3 cross-domain takeaways.',
    'FINDINGS JSON:',
    domainsJson,
    'VERIFICATION CORRECTIONS (refuted/uncertain items to honor):',
    correctionsText || '(none)',
    'Return ONLY the markdown for this section, starting with the "## ' + cluster.title + '" heading.',
  ].join('\n\n')
}

function finalReportPrompt(titlesList, digestJson, verifJson) {
  return [
    'You are the lead author assembling a multi-agent research report titled "OpenCV + Webcam Computer Vision: State of the Art & Research Directions (2026)".',
    PROJECT_CONTEXT,
    'The detailed thematic sections (listed next) are written by other authors and will be appended AFTER your text — DO NOT reproduce their content. A references table and a verification appendix are generated separately — do NOT write them.',
    'Thematic sections that will follow your front matter: ' + titlesList + '.',
    'Write polished markdown with these parts ONLY:',
    '1. "## Executive summary" — 10-14 punchy, actionable bullet takeaways.',
    '2. "## The 2026 landscape" — 2-4 paragraphs on how webcam/edge CV has shifted (foundation models, real-time transformers vs YOLO, promptable/open-vocab, on-device quantization, monocular depth), with inline citations (Author et al., Venue Year).',
    '3. "## Cross-cutting themes" — themes spanning the clusters (real-time on consumer hardware; promptability/zero-shot; data efficiency; privacy/on-device; multi-modal scene understanding).',
    '4. "## Prioritized roadmap for this project" — a markdown table with columns | Priority | Initiative | Why now | Key papers | Target module/script | Effort |. 10-14 rows mapping findings to REAL repo paths (src/models, src/tracking, src/analytics, src/deployment, src/multicam, src/training, scripts/run_*.py). Priority is P0/P1/P2.',
    '5. "## Top open research questions" — 8 concrete questions.',
    '6. "## Conclusion" — one short paragraph.',
    'Compact verified-findings digest (one entry per domain: top method + headline + integration target):',
    digestJson,
    'Verification summary (counts of confirmed/refuted/uncertain headline claims):',
    verifJson,
    'Return ONLY the markdown described above.',
  ].join('\n\n')
}

function criticPrompt(domainList, sectionsDigest) {
  return [
    'You are a COMPLETENESS CRITIC reviewing a multi-agent CV research effort for the OpenCV + webcam platform described in the report.',
    'Domains covered (24): ' + domainList + '.',
    'Condensed synthesized sections:',
    sectionsDigest,
    'Identify: gaps (important sub-topics/methods missed WITHIN the covered domains); missingTopics (entire areas a webcam-CV platform should consider that were not covered — reason about what is absent, e.g. audio-visual, event cameras, gaze, hand tracking, 3D reconstruction, tracking-by-language, etc.); weakestFindings (claims that look thin or under-verified); suggestedNextSteps (concrete follow-up research/experiments). Be specific and critical.',
    'Return ONLY the structured object.',
  ].join('\n\n')
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normTitle(t) { return (t || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim() }
function extractTag(s) { const m = /\b(P\d+)\b/.exec(s || ''); return m ? m[1] : '' }
function chunk(arr, n) {
  const out = []
  for (let i = 0; i < n; i++) out.push([])
  arr.forEach(function (x, i) { out[i % n].push(x) })
  return out.filter(function (c) { return c.length })
}
function compactForSynth(r) {
  return {
    domain: r.domain.title,
    module: r.domain.module,
    overview: r.survey.overview,
    keyTechniques: r.survey.keyTechniques,
    sotaMethods: r.survey.sotaMethods,
    papers: (r.survey.candidatePapers || []).map(function (p) {
      return { title: p.title, authors: p.authors, venue: p.venue, year: p.year, arxivId: p.arxivId, keyClaim: p.keyClaim || p.whyImportant }
    }),
    deepRead: r.card ? {
      paper: r.card.paper && r.card.paper.title,
      method: r.card.method,
      headlineResults: r.card.headlineResults,
      realtime: r.card.realtimeWebcamApplicability,
      integration: r.card.integrationIdea,
      limitations: r.card.limitations,
    } : null,
    openQuestions: r.survey.openQuestions || [],
  }
}

// ---------------------------------------------------------------------------
// Phase 1 + 2 — Survey then Deep-Dive (pipelined, no barrier)
// ---------------------------------------------------------------------------

phase('Survey')
log('Launching ' + DOMAINS.length + ' domain surveys + deep-reads on opus (web-grounded)...')

const pipelined = await pipeline(
  DOMAINS,
  function (d) {
    return agent(surveyPrompt(d), { label: 'survey:' + d.key, phase: 'Survey', model: 'opus', schema: SURVEY_SCHEMA })
  },
  function (survey, d) {
    if (!survey) return null
    return agent(deepDivePrompt(d, survey), { label: 'deep:' + d.key, phase: 'Deep-Dive', model: 'opus', schema: PAPERCARD_SCHEMA })
      .then(function (card) { return { domain: d, survey: survey, card: card } })
  }
)
const domainResults = pipelined.filter(Boolean).filter(function (r) { return r && r.survey })
log('Surveys + deep-reads complete for ' + domainResults.length + '/' + DOMAINS.length + ' domains.')

// ---------------------------------------------------------------------------
// Phase 3 — Adversarial verification of the headline papers (barrier)
// ---------------------------------------------------------------------------

phase('Verify')
const verifyTargets = []
domainResults.forEach(function (r) {
  const p = r.card && r.card.paper
  if (p && p.title) {
    verifyTargets.push({ key: r.domain.key, title: p.title, meta: [p.authors, p.venue, p.year].filter(Boolean).join(', '), claim: (r.card.headlineResults || '').slice(0, 240), arxivId: p.arxivId || '', url: p.url || '' })
  } else {
    const sp = (r.survey.candidatePapers || [])[0]
    if (sp && sp.title) verifyTargets.push({ key: r.domain.key, title: sp.title, meta: [sp.authors, sp.venue, sp.year].filter(Boolean).join(', '), claim: (sp.keyClaim || '').slice(0, 240), arxivId: sp.arxivId || '', url: sp.url || '' })
  }
})
verifyTargets.forEach(function (t, i) { t.tag = 'P' + (i + 1) })

const verifyChunks = chunk(verifyTargets, 4)
log('Adversarially verifying ' + verifyTargets.length + ' headline papers across ' + verifyChunks.length + ' fact-checkers...')
const verifyResults = (await parallel(verifyChunks.map(function (ch, ci) {
  return function () {
    const items = ch.map(function (t) {
      return '[' + t.tag + '] "' + t.title + '" — ' + t.meta + (t.arxivId ? ' (arXiv:' + t.arxivId + ')' : '') + (t.url ? ' ' + t.url : '') + '\n    key claim to check: ' + (t.claim || 'n/a')
    }).join('\n\n')
    return agent(verifyPrompt(items), { label: 'verify:' + (ci + 1), phase: 'Verify', model: 'opus', schema: VERIFY_SCHEMA })
  }
}))).filter(Boolean)

const verdicts = {}
verifyResults.forEach(function (vr) {
  ((vr && vr.results) || []).forEach(function (res) {
    const tag = ((res.tag || '').trim()) || extractTag(res.item)
    if (tag) verdicts[tag] = res
  })
})

let confirmed = 0, refuted = 0, uncertain = 0
const flagged = []
verifyTargets.forEach(function (t) {
  const v = verdicts[t.tag]
  if (!v) { uncertain++; return }
  if (v.verdict === 'confirmed') { confirmed++ }
  else if (v.verdict === 'refuted') { refuted++; flagged.push({ title: t.title, domain: t.key, verdict: 'refuted', evidence: v.evidence || '', correction: v.correction || '' }) }
  else { uncertain++; flagged.push({ title: t.title, domain: t.key, verdict: 'uncertain', evidence: v.evidence || '', correction: v.correction || '' }) }
})
const verificationSummary = { totalChecked: verifyTargets.length, confirmed: confirmed, refuted: refuted, uncertain: uncertain, flagged: flagged }
log('Verification: ' + confirmed + ' confirmed, ' + refuted + ' refuted, ' + uncertain + ' uncertain.')

// ---------------------------------------------------------------------------
// Phase 4 — Thematic synthesis (barrier, one agent per cluster)
// ---------------------------------------------------------------------------

phase('Synthesize')
const correctionsText = flagged.length
  ? flagged.map(function (f) { return '- [' + f.verdict + '] "' + f.title + '" :: ' + (f.correction || f.evidence || '') }).join('\n')
  : ''
const sectionsRaw = await parallel(CLUSTERS.map(function (c) {
  return function () {
    const ds = domainResults.filter(function (r) { return r.domain.cluster === c.key })
    const json = JSON.stringify(ds.map(compactForSynth), null, 1)
    return agent(synthPrompt(c, json, correctionsText), { label: 'synth:' + c.key, phase: 'Synthesize', model: 'opus' })
      .then(function (md) { return { cluster: c, markdown: md } })
  }
}))
const sectionResults = sectionsRaw.filter(Boolean).filter(function (s) { return s && s.markdown })

// ---------------------------------------------------------------------------
// Phase 5 — Lead-author report + completeness critic (parallel)
// ---------------------------------------------------------------------------

phase('Report')
const digest = domainResults.map(function (r) {
  return {
    domain: r.domain.title,
    cluster: r.domain.cluster,
    topMethod: r.card && r.card.paper && r.card.paper.title,
    headline: r.card && r.card.headlineResults,
    integration: (r.card && r.card.integrationIdea) || r.survey.relevanceToProject,
    module: r.domain.module,
  }
})
const titlesList = CLUSTERS.map(function (c) { return c.title }).join('; ')
const sectionsDigest = sectionResults.map(function (s) { return s.markdown.slice(0, 1200) }).join('\n\n---\n\n')
const domainTitles = DOMAINS.map(function (d) { return d.title }).join('; ')

const reportPair = await parallel([
  function () {
    return agent(
      finalReportPrompt(titlesList, JSON.stringify(digest, null, 1), JSON.stringify({ totalChecked: verificationSummary.totalChecked, confirmed: confirmed, refuted: refuted, uncertain: uncertain }, null, 1)),
      { label: 'final-report', phase: 'Report', model: 'opus' }
    )
  },
  function () {
    return agent(criticPrompt(domainTitles, sectionsDigest), { label: 'completeness-critic', phase: 'Report', model: 'opus', schema: CRITIC_SCHEMA })
  },
])
const reportFrontMatter = reportPair[0] || '## Executive summary\n\n_(front-matter generation failed; see thematic sections below.)_'
const critique = reportPair[1] || null

// ---------------------------------------------------------------------------
// Deterministic post-processing: references + paper cards
// ---------------------------------------------------------------------------

const titleVerdict = {}
verifyTargets.forEach(function (t) {
  const v = verdicts[t.tag]
  if (v) titleVerdict[normTitle(t.title)] = v.verdict
})

const refMap = {}
domainResults.forEach(function (r) {
  const collect = (r.survey.candidatePapers || []).slice()
  if (r.card && r.card.paper && r.card.paper.title) collect.push(r.card.paper)
  collect.forEach(function (p) {
    if (!p || !p.title) return
    const k = normTitle(p.title)
    if (!k) return
    if (!refMap[k]) {
      refMap[k] = {
        title: p.title, authors: p.authors || '', venue: p.venue || '', year: p.year || '',
        arxivId: p.arxivId || '', url: p.url || (p.arxivId ? 'https://arxiv.org/abs/' + p.arxivId : ''),
        domains: [r.domain.title], verdict: titleVerdict[k] || '',
      }
    } else {
      if (refMap[k].domains.indexOf(r.domain.title) === -1) refMap[k].domains.push(r.domain.title)
      if (!refMap[k].arxivId && p.arxivId) refMap[k].arxivId = p.arxivId
      if (!refMap[k].url && p.url) refMap[k].url = p.url
      if (!refMap[k].verdict && titleVerdict[k]) refMap[k].verdict = titleVerdict[k]
    }
  })
})
const references = Object.keys(refMap).map(function (k) { return refMap[k] }).sort(function (a, b) {
  return (b.year || 0) - (a.year || 0) || a.title.localeCompare(b.title)
})

const paperCards = domainResults.filter(function (r) { return r.card }).map(function (r) {
  return {
    domain: r.domain.title, module: r.domain.module, paper: r.card.paper, method: r.card.method,
    headlineResults: r.card.headlineResults, realtimeWebcamApplicability: r.card.realtimeWebcamApplicability,
    integrationIdea: r.card.integrationIdea, limitations: r.card.limitations || '', confidence: r.card.confidence || '',
  }
})

const stats = {
  domains: DOMAINS.length,
  domainsCompleted: domainResults.length,
  deepReads: domainResults.filter(function (r) { return r.card }).length,
  verifiers: verifyChunks.length,
  papersChecked: verifyTargets.length,
  uniqueReferences: references.length,
  sections: sectionResults.length,
  agentsApprox: DOMAINS.length * 2 + verifyChunks.length + CLUSTERS.length + 2,
}
log('Done. ' + stats.uniqueReferences + ' unique papers; ' + refuted + ' refuted / ' + uncertain + ' uncertain of ' + verificationSummary.totalChecked + ' headline papers checked. ~' + stats.agentsApprox + ' agents.')

return {
  reportFrontMatter: reportFrontMatter,
  sections: sectionResults.map(function (s) { return { title: s.cluster.title, markdown: s.markdown } }),
  references: references,
  verificationSummary: verificationSummary,
  paperCards: paperCards,
  domainDigest: digest,
  critique: critique,
  stats: stats,
}
