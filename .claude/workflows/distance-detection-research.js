/**
 * distance-detection-research
 * ---------------------------------------------------------------------------
 * A web-grounded, multi-agent research sweep on REAL-TIME OBJECT DETECTION +
 * DISTANCE/RANGE ESTIMATION for this OpenCV/webcam platform. Surveys 10 domains
 * (one cv-researcher each), deep-reads the top method of each into a structured
 * card, adversarially fact-checks the headline claims, synthesizes four thematic
 * sections, and produces a lead-author report + roadmap plus a completeness critique.
 *
 * Pipeline by default: each domain flows survey -> deep-dive -> verify with no
 * barrier; the barrier comes only at synthesis (needs all cards) and report.
 *
 * Run:   Workflow({ name: 'distance-detection-research' })
 *   or:  Workflow({ scriptPath: '.claude/workflows/distance-detection-research.js' })
 *
 * Returns { frontMatter, sections[], cards[], critique, stats }.
 */

export const meta = {
  name: 'distance-detection-research',
  description: 'Web-grounded multi-agent research on real-time object detection + distance/range estimation (mono/stereo/geometric depth, fusion, collision/TTC, calibration, edge) with adversarial fact-check, synthesis, and a repo roadmap',
  whenToUse: 'Deep, cited research on detecting objects and estimating their distance in real time on webcam/edge hardware — produces a report + roadmap for this repo.',
  phases: [
    { title: 'Survey', detail: 'one cv-researcher per domain (10) — web-grounded survey + real papers' },
    { title: 'Deep-Dive', detail: 'deep-read each domain top method into a structured card (10)' },
    { title: 'Verify', detail: 'adversarial fact-check: method real? headline numbers right? (10)' },
    { title: 'Synthesize', detail: 'one cited markdown section per thematic cluster (4)' },
    { title: 'Report', detail: 'lead-author front matter + roadmap, completeness critic (2)' },
  ],
}

// ---------------------------------------------------------------------------
// Shared prompt fragments
// ---------------------------------------------------------------------------

const PROJECT_CONTEXT = [
  'Target repo: a real-time OpenCV + webcam/IP/RTSP computer-vision platform (Python 3.11).',
  'It already has: YOLO/RT-DETR object detection behind a factory, multi-object tracking + re-ID,',
  'a monocular depth subsystem (src/depth/, ONNX Depth-Anything/MiDaS) that writes Detection.depth,',
  'a run_depth.py proximity-alert entrypoint, and multicam geometry (homography + Hungarian).',
  'Research must serve estimating the DISTANCE/RANGE to detected objects in real time on consumer',
  'webcams and edge hardware (CPU/Jetson), and detecting those objects fast enough to be useful.',
].join(' ')

const WEB_RULES = [
  'You MUST ground every claim in live web search before answering — do not rely on memory; the year is 2026.',
  'Use WebSearch and WebFetch (and any mcp__* search tools); if a search tool is not loaded, call ToolSearch with query "select:WebSearch,WebFetch" first.',
  'Prefer primary sources: arxiv.org, openaccess.thecvf.com, paperswithcode.com, official GitHub repos, project pages.',
  'NEVER invent a paper, author, arXiv id, URL, or benchmark number. If you cannot verify a field, leave it blank. Accuracy beats completeness.',
].join(' ')

// ---------------------------------------------------------------------------
// Domains (one survey agent each)
// ---------------------------------------------------------------------------

const DOMAINS = [
  { key: 'realtime-detectors', title: 'Real-time object detectors', focus: 'YOLOv8/v9/v10/v11/YOLO26, RT-DETR/RT-DETRv2, RTMDet, D-FINE, LW-DETR; COCO mAP vs latency, NMS-free/anchor-free, edge variants.' },
  { key: 'mono-metric-depth', title: 'Monocular metric depth', focus: 'Depth Anything V2, Metric3D/Metric3Dv2, UniDepth, ZoeDepth, Depth Pro; metric vs relative depth, zero-shot accuracy (AbsRel, delta1), speed.' },
  { key: 'stereo-depth', title: 'Stereo & multi-view depth', focus: 'RAFT-Stereo, IGEV-Stereo, FoundationStereo; baseline+disparity -> metric range, real-time stereo, hardware needs.' },
  { key: 'geometric-distance', title: 'Geometric monocular distance', focus: 'pinhole model, known-object-size, ground-plane / inverse-perspective-mapping homography, camera height+pitch, vanishing point; closed-form range + error sources.' },
  { key: 'detection-depth-fusion', title: 'Detection-depth fusion', focus: 'sampling depth within a bbox, robust per-object range (median/MAD), RGB-D / frustum fusion; how a distance gets attached to a detection.' },
  { key: 'mono-3d-detection', title: 'Monocular 3D detection & range', focus: 'MonoFlex, FCOS3D, MonoDETR, BEVFormer; 3D bbox + distance for ADAS, KITTI/nuScenes metrics, real-time feasibility.' },
  { key: 'collision-ttc', title: 'Collision / time-to-contact', focus: 'time-to-collision, looming / optical-flow scale change, forward-collision warning, monocular speed estimation.' },
  { key: 'calibration-accuracy', title: 'Calibration & distance accuracy', focus: 'intrinsic/extrinsic calibration, monocular scale ambiguity, depth metrics (AbsRel, RMSE, delta1), dominant error sources, evaluation protocol.' },
  { key: 'edge-deployment', title: 'Edge real-time deployment', focus: 'TensorRT/ONNX/INT8 quantization for detectors AND depth nets, FPS on Jetson/CPU, model sizes, joint detection+depth latency budgets.' },
  { key: 'track-distance', title: 'Temporal distance & tracking', focus: 'smoothing range over tracks, Kalman filtering on distance, speed estimate; stability of per-object distance across frames.' },
]

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const SURVEY_SCHEMA = {
  type: 'object',
  properties: {
    domain: { type: 'string' },
    summary: { type: 'string' },
    methods: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          idea: { type: 'string' },
          citationTitle: { type: 'string' },
          citationUrl: { type: 'string' },
          year: { type: 'number' },
          accuracy: { type: 'string' },
          latencyFps: { type: 'string' },
          hardware: { type: 'string' },
          license: { type: 'string' },
          fitForRepo: { type: 'string' },
        },
        required: ['name', 'idea', 'citationTitle'],
      },
    },
    topMethod: { type: 'string' },
  },
  required: ['domain', 'summary', 'methods', 'topMethod'],
}

const CARD_SCHEMA = {
  type: 'object',
  properties: {
    method: { type: 'string' },
    citationTitle: { type: 'string' },
    citationUrl: { type: 'string' },
    year: { type: 'number' },
    howItWorks: { type: 'string' },
    inputsOutputs: { type: 'string' },
    accuracy: { type: 'string' },
    latencyFps: { type: 'string' },
    hardware: { type: 'string' },
    license: { type: 'string' },
    fitForRepo: { type: 'string' },
    limitations: { type: 'string' },
  },
  required: ['method', 'howItWorks', 'fitForRepo'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    paperReal: { type: 'boolean' },
    numbersPlausible: { type: 'boolean' },
    issues: { type: 'string' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['paperReal', 'numbersPlausible', 'confidence'],
}

const SECTION_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    markdown: { type: 'string' },
    references: {
      type: 'array',
      items: {
        type: 'object',
        properties: { title: { type: 'string' }, url: { type: 'string' }, year: { type: 'number' } },
        required: ['title'],
      },
    },
  },
  required: ['title', 'markdown'],
}

const REPORT_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    executiveSummary: { type: 'string' },
    roadmap: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item: { type: 'string' },
          rationale: { type: 'string' },
          module: { type: 'string' },
          effort: { type: 'string', enum: ['S', 'M', 'L'] },
          priority: { type: 'string', enum: ['P0', 'P1', 'P2'] },
        },
        required: ['item', 'rationale', 'module'],
      },
    },
  },
  required: ['title', 'executiveSummary', 'roadmap'],
}

const CRITIQUE_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'string' } },
    unverifiedClaims: { type: 'array', items: { type: 'string' } },
    missingDomains: { type: 'array', items: { type: 'string' } },
    overallConfidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['gaps', 'overallConfidence'],
}

// ---------------------------------------------------------------------------
// Phases 1-3: survey -> deep-dive -> verify, pipelined per domain (no barrier)
// ---------------------------------------------------------------------------

phase('Survey')
const perDomain = await pipeline(
  DOMAINS,
  // stage 1 — survey the domain
  d => agent(
    [
      `Survey the 2024-2026 state of the art for: ${d.title}.`,
      `Focus: ${d.focus}`,
      PROJECT_CONTEXT,
      WEB_RULES,
      'Return 3-6 methods, each with a real citation, reported accuracy and latency/FPS (with the hardware it was measured on), license, and a one-line fit-for-this-repo note.',
      `Also name a single best "topMethod" for real-time webcam/edge use. Set domain to "${d.key}".`,
    ].join('\n'),
    { label: `survey:${d.key}`, phase: 'Survey', agentType: 'cv-researcher', schema: SURVEY_SCHEMA },
  ),
  // stage 2 — deep-read the top method into a card
  (survey, d) => agent(
    [
      `Deep-read the single most relevant method for "${d.title}": ${survey && survey.topMethod ? survey.topMethod : '(pick the strongest from the survey below)'}.`,
      'Produce a structured card: how it works, inputs/outputs, accuracy (with metric+dataset), latency/FPS + hardware, license, fit for this repo (which module it would extend), and limitations.',
      PROJECT_CONTEXT,
      WEB_RULES,
      `Survey context: ${JSON.stringify(survey || {})}`,
    ].join('\n'),
    { label: `deepdive:${d.key}`, phase: 'Deep-Dive', agentType: 'cv-researcher', schema: CARD_SCHEMA },
  ).then(card => ({ domain: d.key, title: d.title, survey, card })),
  // stage 3 — adversarially fact-check the card
  item => agent(
    [
      'Adversarially fact-check this research card. Try to REFUTE it.',
      'Find the primary source: is the method/paper real? Are the headline accuracy/latency numbers plausible and correctly attributed (right dataset, right hardware)?',
      'Default to skepticism; if you cannot confirm a number, say so.',
      WEB_RULES,
      `Card: ${JSON.stringify(item.card)}`,
    ].join('\n'),
    { label: `verify:${item.domain}`, phase: 'Verify', agentType: 'cv-researcher', schema: VERDICT_SCHEMA },
  ).then(verdict => ({ ...item, verdict })),
)

const cards = perDomain.filter(Boolean)
log(`surveyed + deep-read + verified ${cards.length}/${DOMAINS.length} domains`)

// ---------------------------------------------------------------------------
// Phase 4: synthesize thematic sections (barrier — needs all cards)
// ---------------------------------------------------------------------------

const CLUSTERS = [
  { key: 'detection', title: 'Real-time object detection', domains: ['realtime-detectors', 'edge-deployment'] },
  { key: 'distance', title: 'Distance & depth estimation methods', domains: ['mono-metric-depth', 'stereo-depth', 'geometric-distance', 'mono-3d-detection'] },
  { key: 'fusion', title: 'Attaching distance to detections: fusion, collision & tracking', domains: ['detection-depth-fusion', 'collision-ttc', 'track-distance'] },
  { key: 'accuracy', title: 'Calibration, accuracy & edge deployment', domains: ['calibration-accuracy', 'edge-deployment'] },
]

phase('Synthesize')
const sections = await parallel(CLUSTERS.map(c => () => {
  const relevant = cards.filter(x => c.domains.includes(x.domain))
  return agent(
    [
      `Write a cited markdown section titled "${c.title}" for a research report.`,
      'Synthesize across the verified research cards below: compare methods, state which are real-time on webcam/edge, and be concrete about accuracy and latency with the hardware noted.',
      'Use inline references like [Method/Author, Year](url). Do NOT invent anything not present in the cards or independently verifiable. Prefer tables for method comparisons.',
      PROJECT_CONTEXT,
      `Verified cards: ${JSON.stringify(relevant.map(x => ({ domain: x.domain, card: x.card, verdict: x.verdict })))}`,
    ].join('\n'),
    { label: `synth:${c.key}`, phase: 'Synthesize', agentType: 'cv-researcher', schema: SECTION_SCHEMA },
  )
}))
const goodSections = sections.filter(Boolean)
log(`synthesized ${goodSections.length}/${CLUSTERS.length} sections`)

// ---------------------------------------------------------------------------
// Phase 5: report (lead author + completeness critic), barrier
// ---------------------------------------------------------------------------

phase('Report')
const [report, critique] = await parallel([
  () => agent(
    [
      'You are the lead author. Using the synthesized sections and the cards, write the report front matter:',
      'a title, an executive summary (what works TODAY for real-time detection + distance on a webcam/edge), and a prioritized ROADMAP for THIS repo.',
      'Each roadmap item: what to build, why, which module it extends (e.g. src/depth/, src/models/, a new src/distance/ geometric helper, run_depth.py), effort S/M/L, priority P0/P1/P2.',
      'Favor additive, config-gated, lazy-import-friendly changes that fit the existing depth + detection seams. Be concrete and buildable.',
      PROJECT_CONTEXT,
      `Section titles: ${JSON.stringify(goodSections.map(s => s.title))}`,
      `Card fit notes: ${JSON.stringify(cards.map(x => ({ domain: x.domain, method: x.card && x.card.method, fit: x.card && x.card.fitForRepo })))}`,
    ].join('\n'),
    { label: 'report:lead', phase: 'Report', agentType: 'cv-researcher', schema: REPORT_SCHEMA },
  ),
  () => agent(
    [
      'You are a completeness critic for this research. Identify gaps honestly:',
      'what is missing, which claims remain unverified or low-confidence, and which relevant domains were not covered (or deserve a deeper pass).',
      `Domains covered: ${JSON.stringify(DOMAINS.map(d => d.title))}`,
      `Verification verdicts: ${JSON.stringify(cards.map(x => ({ domain: x.domain, verdict: x.verdict })))}`,
    ].join('\n'),
    { label: 'report:critic', phase: 'Report', agentType: 'cv-researcher', schema: CRITIQUE_SCHEMA },
  ),
])

return {
  frontMatter: report,
  sections: goodSections,
  cards: cards.map(x => ({ domain: x.domain, title: x.title, card: x.card, verdict: x.verdict, methods: x.survey && x.survey.methods })),
  critique,
  stats: { domains: DOMAINS.length, verified: cards.length, sections: goodSections.length },
}
