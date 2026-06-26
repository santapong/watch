# Authoring workflows (and workflow skills)

How to write a multi-agent **Workflow** for this repo, and how to package one as a named
workflow or a `/slash` skill. Pair this with [`template.js`](./template.js).

## Two artifacts, two ways to invoke

| Artifact | Lives in | Invoked by | Use for |
|----------|----------|------------|---------|
| **Workflow script** | `.claude/workflows/<name>.js` | `Workflow({ name: '<name>' })` or `Workflow({ scriptPath })` | the deterministic orchestration itself |
| **Skill** | `.claude/skills/<name>/SKILL.md` | `/<name>` (Skill tool) | a procedure an agent follows; *may* call the `Workflow` tool |

The `/workflow` skill here is a **skill** that helps you author/run **workflow scripts**.

## Anatomy of a workflow script

```js
export const meta = { /* PURE LITERAL */
  name: 'my-workflow',
  description: 'one line, shown in the run dialog',
  whenToUse: 'optional — shown in the workflow list',
  phases: [ { title: 'Find' }, { title: 'Verify' } ],  // match phase() titles exactly
}

// body runs in async scope — use await directly
phase('Find')
const found = await agent('…', { schema: SCHEMA })
```

`meta` **must be a pure literal** — no variables, function calls, spreads, or template
interpolation. Required: `name`, `description`.

## The hooks

| Hook | What it does |
|------|--------------|
| `agent(prompt, opts?)` | spawn a subagent. Returns its text, or the validated object if `opts.schema` is set. `opts`: `label`, `phase`, `schema`, `model`, `effort`, `agentType`, `isolation:'worktree'`. Returns `null` if it dies/skips — `.filter(Boolean)`. |
| `pipeline(items, …stages)` | run each item through all stages independently — **no barrier**. The default. Stage cb gets `(prevResult, originalItem, index)`. |
| `parallel(thunks)` | run thunks concurrently and **await all** (a barrier). Failures become `null`. |
| `phase(title)` / `log(msg)` | progress grouping / a narrator line to the user. |
| `args` / `budget` / `workflow(name, args?)` | input value; token budget (`budget.total`, `.remaining()`); run another workflow inline (one level deep). |

## Rules & gotchas

- **JavaScript, not TypeScript** — no type annotations/interfaces/generics.
- **No** `Date.now()`, `Math.random()`, or argless `new Date()` (they break resume). Vary by
  index; stamp time after the workflow returns or via `args`.
- **Default to `pipeline`.** Reach for a `parallel` barrier only when a stage needs *all*
  prior results at once (dedup/merge, early-exit on zero, cross-item comparison). "I need to
  flatten/filter first" is not a reason — do it inside a pipeline stage.
- **Structured output:** pass a JSON-Schema `schema` and the agent returns a validated object
  (it retries on mismatch) — no parsing.
- **Specialists:** `agentType: 'cv-researcher' | 'cv-implementer' | 'cv-reviewer'` pulls the
  scoped agents in `.claude/agents/` (right tools + golden rules baked in).
- **Parallel file edits:** add `isolation: 'worktree'` so agents don't clobber each other
  (expensive — only when they actually mutate files concurrently).
- **Concurrency** is capped (~CPU-2, max 16); pass as many items as you like, they queue.

## Patterns (compose freely)

- **Adversarial verify** — N skeptics per finding, prompted to *refute*; keep if a majority
  can't. (See `template.js`.)
- **Judge panel** — N independent attempts from different angles → parallel scorers →
  synthesize the winner, grafting runners-up.
- **Loop-until-dry** — keep spawning finders until K consecutive rounds find nothing new;
  dedup against everything *seen*, not just confirmed.
- **Multi-modal sweep** — agents each search a different way (by-file, by-symbol, by-entity).
- **Completeness critic** — a final agent asking "what's missing?"; its output is the next
  round.
- **Loop-until-budget** — `while (budget.total && budget.remaining() > 50_000) { … }`.

## Iterate & resume

- Every `Workflow` run persists its script and returns a `scriptPath` and `runId`.
- Edit that file, re-run with `{ scriptPath }`. To resume after a stop/edit, add
  `{ resumeFromRunId }` — the unchanged prefix returns cached results; the first changed
  call onward re-runs. Same script + same args ⇒ 100% cache hit.

## Promote a script to a named workflow

1. Save it as `.claude/workflows/<name>.js` with a matching `meta.name`.
2. Run it anywhere with `Workflow({ name: '<name>' })`; parameterize via `args`.
3. Example in this repo: `.claude/workflows/opencv-webcam-cv-research.js`.

## Wrap a workflow in a `/slash` skill

Create `.claude/skills/<name>/SKILL.md`:

```markdown
---
name: my-review
description: When to use this — be specific; this is how it's auto-selected.
---

# /my-review
Steps the agent follows. To run the orchestration, call the Workflow tool:
`Workflow({ name: 'review-diff' })` (or `{ scriptPath: '.claude/skills/.../template.js' }`),
then read the returned object and act on it.
```

Keep `description` precise — it drives when the skill is suggested. Put any long reference
(like this file) alongside `SKILL.md` and link to it rather than inlining.
