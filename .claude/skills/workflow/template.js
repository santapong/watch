/**
 * template — a starter multi-agent workflow for this repo. Copy, rename, adapt.
 *
 * What it does: reviews the current git diff across several dimensions in parallel,
 * then ADVERSARIALLY verifies each finding — a finding survives only if a majority of
 * independent skeptics fail to refute it. Pipeline by default: each dimension's findings
 * start verifying the moment that dimension finishes (no barrier between stages).
 *
 * Run as-is:   Workflow({ scriptPath: '.claude/skills/workflow/template.js' })
 * Or register: move to .claude/workflows/review-diff.js → Workflow({ name: 'review-diff' })
 */

export const meta = {
  // meta MUST be a pure literal — no variables, calls, spreads, or interpolation.
  name: 'review-diff',
  description: 'Review the current diff across dimensions and adversarially verify each finding',
  whenToUse: 'Before merging a change, to surface only findings that survive a refute panel.',
  phases: [
    { title: 'Review' },
    { title: 'Verify' },
  ],
}

// --- structured-output schemas: force agents to return validated JSON ---
const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'number' },
          severity: { type: 'string', enum: ['blocker', 'should-fix', 'nit'] },
          detail: { type: 'string' },
        },
        required: ['title', 'file', 'severity', 'detail'],
      },
    },
  },
  required: ['findings'],
}
const VERDICT = {
  type: 'object',
  properties: { real: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['real', 'reason'],
}

// --- the review dimensions (adapt these to your task) ---
const DIMENSIONS = [
  { key: 'correctness', prompt: 'Review the current git diff for CORRECTNESS bugs only (logic, shapes/axes, None/empty handling, off-by-one, mask/threshold inversions).' },
  { key: 'invariants',  prompt: 'Review the current git diff for violations of CLAUDE.md invariants: top-level heavy imports, changed defaults, untested pure logic, dishonest "validated" claims.' },
  { key: 'simplicity',  prompt: 'Review the current git diff for duplication an existing helper already covers, and for needless complexity.' },
]

const VERIFIERS = 3 // skeptics per finding

phase('Review')
const results = await pipeline(
  DIMENSIONS,
  // stage 1 — one cv-reviewer per dimension; receives (item, item, index)
  d => agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', agentType: 'cv-reviewer', schema: FINDINGS }),
  // stage 2 — verify this dimension's findings; receives (prevResult, originalItem, index)
  (review, d) => parallel((review?.findings || []).map(f => () =>
    parallel(Array.from({ length: VERIFIERS }, (_, i) => () =>
      agent(
        'Try to REFUTE this review finding. Default to real=false if uncertain.\n' +
        `Finding: ${f.title}\nFile: ${f.file}:${f.line || '?'}\nDetail: ${f.detail}`,
        { label: `verify:${d.key}:${i}`, phase: 'Verify', agentType: 'cv-reviewer', schema: VERDICT },
      )
    )).then(votes => {
      const real = votes.filter(Boolean).filter(v => v.real).length > VERIFIERS / 2
      return { ...f, dimension: d.key, confirmed: real }
    })
  )),
)

const confirmed = results.flat().filter(Boolean).filter(f => f.confirmed)
log(`confirmed ${confirmed.length} finding(s) across ${DIMENSIONS.length} dimensions`)
return { confirmed }
