#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Setup script for Codex sandbox:
# 1. Guarantees a modern Python (>=3.8). If the base image only has 3.7,
#    we download a standalone 3.11 and create a virtualenv with `uv`.
# 2. Installs all project dependencies (runtime + dev) via `uv pip`.
# -----------------------------------------------------------------------------

# 1) Ensure `uv` (Rust‑based package manager with embedded Python builds) exists
if ! command -v uv &>/dev/null; then
  curl -Ls https://astral.sh/uv/install | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# 2) If system Python is <3.8, bootstrap Python 3.11 into .venv
if python - <<'PY'
import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)
PY
then
  echo "✅ Using system $(python -V)"
else
  echo "⏬ System Python too old; installing standalone 3.11 via uv"
  uv venv --python 3.11 .venv
  source .venv/bin/activate
fi

# 3) Install dependencies (system Python vs virtualenv)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # Installing into the system interpreter – tell uv to allow it
  uv pip install --system -r requirements.txt -r requirements-dev.txt
else
  uv pip install -r requirements.txt -r requirements-dev.txt
fi

# ---------------------------------------------------------------------------
# 4) Fallback: offline wheel‑house install
#
# Codex sandbox sometimes runs with outbound network disabled. In that case
# the previous uv‑pip step fails with “No route to host”.  If we detect that
# failure – or if pytest is still missing – try to install from a local
# wheel‑house directory committed to the repo:  .wheels/
# ---------------------------------------------------------------------------
set +e
python - <<'PY'
import importlib.util, sys
missing = [m for m in ("pytest", "httpx") if importlib.util.find_spec(m) is None]
sys.exit(0 if not missing else 1)
PY
need_offline=$?
set -e

if [[ "$need_offline" -ne 0 ]]; then
  wheel_dir="$(dirname "$0")/../.wheels"
  if [[ -d "$wheel_dir" ]]; then
    echo "🔄  Network install failed; installing from local wheel‑house"
    uv pip install --no-index --find-links "$wheel_dir" -r requirements.txt -r requirements-dev.txt
  else
    echo "❌ Dependencies missing and .wheels/ not found.  Aborting."
    exit 1
  fi
fi