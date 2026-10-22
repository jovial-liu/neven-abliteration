#!/usr/bin/env bash
# Put the project back to a fresh-checkout state so you can run everything from scratch.
#   bash reset.sh          # remove .venv, the saved edited model, run outputs, caches
#   bash reset.sh --full   # also delete the downloaded model + datasets (~3 GB; re-downloads on next setup)
set -euo pipefail
cd "$(dirname "$0")"

rm -rf .venv .pytest_cache qwen-abliterated qwen-abliterated-healed dpo_pairs.jsonl
find . -name __pycache__ -type d -prune -exec rm -rf {} +
echo "removed: .venv, qwen-abliterated, qwen-abliterated-healed, dpo_pairs.jsonl, caches"

if [ "${1:-}" = "--full" ]; then
  HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  rm -rf "$HUB/models--Qwen--Qwen2.5-1.5B-Instruct" \
         "$HUB/datasets--mlabonne--harmful_behaviors" \
         "$HUB/datasets--mlabonne--harmless_alpaca"
  echo "removed: downloaded model + datasets from $HUB"
fi

echo "Next: bash setup.sh"
