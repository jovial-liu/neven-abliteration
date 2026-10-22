#!/usr/bin/env bash
# One-shot setup for a fresh machine (macOS / Linux). Run from the repo root:
#   bash setup.sh
# Creates .venv, installs deps, runs the tests, and pre-downloads the model so
# the first run doesn't stall on the network. The prompt datasets ship with the
# repo under data/.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. On a Mac run:  xcode-select --install   (or install from python.org), then re-run." >&2
  exit 1
fi
echo "==> Python: $($PY --version)"

if [ ! -d .venv ]; then
  echo "==> Creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

echo "==> Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet pytest
pip install --quiet -r requirements.txt

echo "==> Running tests"
if ! python -m pytest -q tests; then
  echo
  echo "WARNING: some tests failed (see above). Continuing with setup; the script may still work." >&2
fi

echo "==> Pre-downloading the model (one-off, ~3 GB)"
python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-1.5B-Instruct"
AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name)
print("cached OK")
EOF

cat <<'EOF'

Done. Next:
  . .venv/bin/activate
  python abliterate.py                                     # abliterate (~90 s on Apple silicon); saves the edited model
  python chat.py --compare                                 # chat with it: original vs edited (/q to quit)
  bash reset.sh                                            # start over from scratch
EOF
