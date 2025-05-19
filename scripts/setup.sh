#!/usr/bin/env bash
set -euo pipefail

# Use pip if uv is unavailable
USE_PIP=false

# -----------------------------------------------------------------------------
# Setup script for Codex sandbox:
# 1. Guarantees a modern Python (>=3.8). If the base image only has 3.7,
#    we download a standalone 3.11 and create a virtualenv with `uv`.
# 2. Installs all project dependencies (runtime + dev) via `uv pip`.
# -----------------------------------------------------------------------------

# 1) Ensure `uv` (Rust‑based package manager with embedded Python builds) exists
if ! command -v uv &>/dev/null; then
  echo "⏬ Installing uv"
  set +e
  curl -Ls https://astral.sh/uv/install | sh
  curl_status=$?
  set -e
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v uv &>/dev/null; then
    echo "⚠️  uv unavailable; falling back to pip"
    USE_PIP=true
  fi
fi



# 2) If system Python is <3.8, bootstrap Python 3.11 into .venv
if python - <<'PY'
import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)
PY
then
  echo "✅ Using system $(python -V)"
else
  if [ "$USE_PIP" = false ]; then
    echo "⏬ System Python too old; installing standalone 3.11 via uv"
    uv venv --python 3.11 .venv
    source .venv/bin/activate
  else
    echo "⚠️  System Python too old and uv unavailable; using system Python"
  fi
fi

# 3) Install dependencies (system Python vs virtualenv)
set +e
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [ "$USE_PIP" = true ]; then
    pip install -r requirements.txt -r requirements-dev.txt
  else
    # Installing into the system interpreter – tell uv to allow it
    uv pip install --system -r requirements.txt -r requirements-dev.txt
  fi
else
  if [ "$USE_PIP" = true ]; then
    pip install -r requirements.txt -r requirements-dev.txt
  else
    uv pip install -r requirements.txt -r requirements-dev.txt
  fi
fi
install_status=$?
set -e

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
if [[ "$install_status" -ne 0 ]]; then
  need_offline=1
fi
set -e

if [[ "$need_offline" -ne 0 ]]; then
  wheel_dir="$(dirname "$0")/../.wheels"
  if [[ -d "$wheel_dir" ]]; then
    echo "🔄  Network install failed; installing from local wheel‑house"
    if [ "$USE_PIP" = true ]; then
      pip install --no-index --find-links "$wheel_dir" -r requirements.txt -r requirements-dev.txt || true
      pip install --no-index --find-links "$wheel_dir" pytest || true
    else
      if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        uv pip install --system --no-index --find-links "$wheel_dir" -r requirements.txt -r requirements-dev.txt || true
        uv pip install --system --no-index --find-links "$wheel_dir" pytest || true
      else
        uv pip install --no-index --find-links "$wheel_dir" -r requirements.txt -r requirements-dev.txt || true
        uv pip install --no-index --find-links "$wheel_dir" pytest || true
      fi
    fi
    python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("pytest") else 1)
PY
    have_pytest=$?
    if [[ "$have_pytest" -ne 0 ]]; then
      echo "❌ pytest not available after offline install"
      exit 1
    fi
  else
    echo "❌ Dependencies missing and .wheels/ not found.  Aborting."
    exit 1
  fi
fi

# Install project path into site-packages for import
site_dir=$(python -c "import site, sys; print(site.getsitepackages()[0])")
echo "$(pwd)" > "$site_dir/generate_cards.pth"
