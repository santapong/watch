#!/usr/bin/env bash
# SessionStart hook — make the torch-free unit suite runnable the moment a session
# starts (fresh clone / resume / clear), so agents can verify changes immediately.
#
# Design: idempotent (checks before installing), non-fatal (always exits 0 so a slow
# mirror never blocks the session), and slim (installs only the torch-free test set
# that CI uses — seconds, not the ~2 GB ML stack). See docs/HARNESS.md.
set -u

SLIM="pytest numpy pyyaml pillow requests opencv-python-headless scikit-learn"

# Fast path: if the core test deps already import, there's nothing to do.
if python -c "import pytest, numpy, cv2, yaml, PIL, sklearn" >/dev/null 2>&1; then
  echo "[harness] slim test deps present — 'python -m pytest -q' is ready."
  exit 0
fi

echo "[harness] installing slim test deps (torch-free, mirrors CI)…"
if pip install --quiet --disable-pip-version-check $SLIM >/dev/null 2>&1; then
  echo "[harness] ready: python -m pytest -q"
else
  echo "[harness] WARN: slim dep install failed; run 'pip install $SLIM' manually."
fi
exit 0
