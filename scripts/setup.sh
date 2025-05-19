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