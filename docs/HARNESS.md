# Harness Engineering

The **harness** is the operating environment that lets AI agents develop this repo
reliably and repeatably. The [ADLC](./ADLC.md) is *what* we do; the harness is *how the
agent is set up to do it well*. Harness engineering = designing this environment so good
outcomes are the path of least resistance and the [golden rules](../CLAUDE.md#golden-rules-the-invariants-ci-enforces)
are hard to break.

## The layers

```
┌─────────────────────────────────────────────────────────────┐
│  CLAUDE.md            always-loaded contract + project map     │  context
├─────────────────────────────────────────────────────────────┤
│  .claude/settings.json   permissions allowlist + hooks         │  policy
│  .claude/hooks/          SessionStart readiness (slim deps)     │  automation
├─────────────────────────────────────────────────────────────┤
│  .claude/agents/         scoped subagent roles                  │  who
│  .claude/skills/         reusable procedures (/workflow, …)     │  how
│  .claude/workflows/      deterministic multi-agent scripts      │  scale
├─────────────────────────────────────────────────────────────┤
│  tests/conftest.py       stub heavy deps → src imports torch-free │ test seam
│  .github/workflows/ci.yml  slim, torch-free gate                │  enforcement
└─────────────────────────────────────────────────────────────┘
```

Each layer answers one question: **context** (what the agent always knows), **policy**
(what it may do without asking), **automation** (what runs on its own), **who/how/scale**
(which agent, which procedure, how wide), and **enforcement** (what blocks a bad change).

## 1 · Context — `CLAUDE.md`

Loaded into every agent automatically. It carries the project map *and* the invariants, so
an agent doesn't have to rediscover them. The highest-leverage harness artifact: most
"agent did the wrong thing" problems are a missing line in `CLAUDE.md`. Keep it current and
high-signal; link out to `ADLC.md` / `HARNESS.md` rather than inlining everything.

## 2 · Policy — `.claude/settings.json`

- **`permissions.allow`** — read-only and routine commands (pytest, python, git status/diff,
  the slim `pip install`) are pre-approved to cut prompt friction. Anything mutating or
  outward-facing still prompts.
- **`hooks`** — shell commands the *harness* runs at lifecycle events (not the model). This
  is how you enforce behavior deterministically instead of hoping the model remembers.

> Edit settings via the `/update-config` skill (it knows the schema) or carefully by hand;
> keep the allowlist conservative — prefer prompting over a too-broad rule.

## 3 · Automation — `.claude/hooks/`

`session-start.sh` runs on **SessionStart** (fresh clone, resume, clear). It idempotently
installs the slim test deps so the suite can run immediately in an ephemeral cloud session,
then prints a readiness line. It is intentionally:

- **idempotent** — checks before installing; safe to run every session;
- **non-fatal** — always exits 0 so a slow mirror never blocks the session;
- **slim** — installs only the torch-free test set, matching CI (≈ seconds, not GB).

This is the [`session-start-hook`](https://code.claude.com/docs) pattern: guarantee
"tests and linters can run" the moment a web/cloud session starts.

## 4 · Who — `.claude/agents/`

Scoped subagent definitions give the orchestrator specialists with the right tools and a
role-specific system prompt:

| Agent | Tools | Role |
|-------|-------|------|
| `cv-implementer` | edit/test | implement one roadmap item, golden-rules-compliant + tested |
| `cv-reviewer` | read-only | adversarial review against the invariants |
| `cv-researcher` | web + read | web-grounded research; never invents sources |

Spawn via the `Agent` tool (`subagent_type: "cv-implementer"`) or inside a `Workflow`
(`agentType: "cv-reviewer"`). Read-only roles (reviewer/researcher) deliberately lack edit
tools so they cannot drift into making changes.

## 5 · How — `.claude/skills/`

Skills are reusable, invocable procedures (`/name`). The flagship here is **`/workflow`**
(`.claude/skills/workflow/`), which authors and runs a project-conventioned multi-agent
workflow for a task. Skills encode *how we do a recurring thing* so it's one command, not a
re-explanation each time. See the skill's own README for authoring guidance.

## 6 · Scale — `.claude/workflows/`

Deterministic JavaScript scripts that orchestrate many subagents with real control flow
(loops, fan-out, adversarial verify, synthesis). Use a workflow when one context can't hold
the work or when independent perspectives raise confidence: research sweeps, broad reviews,
migrations. `opencv-webcam-cv-research.js` is the worked example. Authoring details live in
the `/workflow` skill.

## 7 · Test seam — `tests/conftest.py`

The keystone of the **import-safety contract**. It stubs `supervision`, `transformers`,
`streamlit`, and `ultralytics` so `src` imports with only the slim deps. Combined with the
rule that heavy runtimes are imported *lazily inside functions*, this lets the whole unit
suite run torch-free in seconds. New module needs a heavy dep? Import it lazily and, if it's
imported at collection time anywhere, add a stub here.

## 8 · Enforcement — `.github/workflows/ci.yml`

The gate that makes the invariants real. It installs **only** the slim deps and runs
`pytest -q`. If someone adds a top-level `import torch`, CI fails to import and goes red —
the rule enforces itself. Fast (~2 GB lighter than a full ML install) so every push is
checked cheaply.

---

## The import-safety contract (why it all hangs together)

This single contract is what keeps the harness fast and the suite trustworthy:

1. **Lazy imports** — heavy runtimes imported inside the function that uses them.
2. **Pure core** — model-independent logic (math, pre/post-proc, buffering, factory) split
   out so it can be unit-tested.
3. **Mock the model** — tests inject/mock the heavy forward; the pure core is asserted for
   real.
4. **conftest stubs + slim CI** — collection never needs the heavy deps.

Result: 252 tests run torch-free in ~2 s; CI is cheap; and "did the model actually run?"
stays an honest, separate question answered on a GPU box.

## Extending the harness — checklist

- **New module with a heavy dep?** Lazy-import it; split a pure core; add mocked tests; add
  a `requirements-*.txt` entry; stub in `conftest.py` if imported at collection.
- **New recurring multi-agent task?** Add a `.claude/workflows/*.js` script and/or a skill.
- **New specialist?** Add a `.claude/agents/*.md` with the minimum tools it needs.
- **Want a behavior enforced, not remembered?** Add a hook in `settings.json`, not a note.
- **Agent keeps missing a rule?** Add one line to `CLAUDE.md`.

## Map of the harness in this repo

| File | Layer |
|------|-------|
| `CLAUDE.md` | context |
| `.claude/settings.json` | policy (permissions + hooks) |
| `.claude/hooks/session-start.sh` | automation (readiness) |
| `.claude/agents/{cv-implementer,cv-reviewer,cv-researcher}.md` | who |
| `.claude/skills/workflow/` | how (the `/workflow` skill) |
| `.claude/workflows/opencv-webcam-cv-research.js` | scale (example workflow) |
| `tests/conftest.py` | test seam |
| `.github/workflows/ci.yml` | enforcement |
| `docs/ADLC.md` | the process this harness serves |
