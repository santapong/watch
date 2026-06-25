---
name: workflow
description: Author and run a project-conventioned multi-agent Workflow for a task in this repo — research sweeps, broad reviews, migrations, or N-way design. Use when one agent context can't hold the work, or when independent/adversarial perspectives would raise confidence. Not for single-file lookups or trivial edits.
---

# /workflow — orchestrate a multi-agent workflow

This skill helps you decide on, author, and run a deterministic multi-agent **Workflow**
for a task in this computer-vision repo, following the conventions in
[`docs/ADLC.md`](../../../docs/ADLC.md) and [`docs/HARNESS.md`](../../../docs/HARNESS.md).

A workflow is the right tool when the work is **wide** (decompose + cover in parallel),
needs **confidence** (independent perspectives, adversarial verification), or is **bigger
than one context** (migrations, audits, sweeps). For a single fact or a one-file edit, just
use the `Agent` tool or do it inline — don't reach for a workflow.

## Procedure

1. **Justify it.** If the task is a lookup or a small edit, stop — use an `Agent` or do it
   directly. Otherwise name *why* a workflow helps: scale, confidence, or scope.
2. **Scout inline first (hybrid).** Discover the work-list before orchestrating: list the
   files, find the call-sites, scope the diff. You don't need to know the shape before the
   *task* — only before the *orchestration step*.
3. **Pick the shape.**
   - `pipeline(items, stageA, stageB, …)` — **default.** Each item flows through all stages
     independently; no barrier. Wall-clock = slowest single chain.
   - `parallel(thunks)` — a **barrier**; use only when a stage genuinely needs *all* prior
     results at once (dedup/merge, early-exit on zero, "compare against the others").
   - Loops — until-count, until-dry (K empty rounds), or until-budget.
4. **Use this repo's specialists.** Spawn agents with `agentType: 'cv-researcher' |
   'cv-implementer' | 'cv-reviewer'` (defined in `.claude/agents/`) so each carries the
   right tools and the golden rules. Force structured returns with a `schema`.
5. **Verify adversarially.** For findings/claims, add a verify stage that *refutes* (≥
   majority must clear it). Give verifiers distinct lenses when failure modes differ.
6. **Author from the template.** Copy `template.js`, fill `meta` (a **pure literal**), wire
   the stages. Iterate by editing the saved script and re-running with `{scriptPath}`.
7. **Run and stay in the loop.** Call the `Workflow` tool; read the returned object; decide
   the next phase yourself. For multi-phase work run several workflows in sequence rather
   than one mega-script.

## Reminders that bind any code agent you spawn

Any agent that writes code here obeys `CLAUDE.md`: **lazy heavy imports**, **pure core +
mocked model in tests**, **additive + config-gated**, **`pytest -q` green torch-free**. Put
these in the agent prompt (or use `agentType: 'cv-implementer'`, which already encodes them).

## Recipes mapped to this repo

| Goal | Shape |
|------|-------|
| **Understand** a subsystem | `parallel` readers over modules → structured map |
| **Research** a technique | the `cv-researcher` fan-out (see `.claude/workflows/opencv-webcam-cv-research.js`) |
| **Review** a diff | `pipeline(dimensions, review, verify)` — each finding refuted by a panel |
| **Design** N approaches | judge panel: N attempts → parallel scorers → synthesize winner |
| **Migrate** many call-sites | `pipeline(sites, transform, verify)` with `isolation:'worktree'` |

## Authoring details

Full reference for writing workflow scripts **and** turning one into a named skill/command:
[`AUTHORING.md`](./AUTHORING.md). Starter script: [`template.js`](./template.js).

> Cost note: workflows can spawn many agents and spend real tokens. Only run one when the
> user has opted into multi-agent orchestration; scale the fleet to the ask (a few finders
> for "any bugs?", a larger pool + 3–5-vote verify for "thoroughly audit this").
