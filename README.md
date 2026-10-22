# Abliteration

Remove refusal behaviour from an instruct model by ablating the "refusal
direction" from its weights ([Arditi et al., NeurIPS 2024](https://arxiv.org/abs/2406.11717);
popularised by Maxime Labonne's
["Uncensor any LLM with abliteration"](https://huggingface.co/blog/mlabonne/abliteration)).

The pipeline in `abliterate.py`:

1. Run the model on harmful and harmless prompts and record the residual-stream
   activation at the last token for every layer.
2. Per layer, the refusal direction is the normalised difference of means
   (harmful minus harmless).
3. Shortlist candidate layers (40–80 % depth) with a cheap hook-based preview
   scored on *fresh* probe prompts (never the eval set).
4. Try real edits: each shortlisted layer is gated with a full-strength (α = 1)
   permanent edit verified on the probes — hooks and weight edits disagree below
   α = 1, so nothing is decided by simulation — and passing layers are walked
   down to the weakest strength that still removes every probe refusal. The
   winner (fewest refusals, then most real answers, then weakest α, then
   shallowest layer) is re-applied and left in place.
5. The edit orthogonalises every weight that writes to the residual stream
   (`embed_tokens`, every `o_proj` and `down_proj`) against that direction:
   `W' = W − α r rᵀ W`. No training, no hooks left behind.
6. Optional `heal.py`: DPO on benign preference pairs (chosen = original
   answer, rejected = edited answer) with LoRA on the matrices that only
   *read* from the residual stream, so the orthogonalised writers stay
   orthogonal and refusals cannot come back through the adapter.

Default target is `Qwen/Qwen2.5-1.5B-Instruct`; any Llama/Qwen/Mistral/Gemma-style
HF model works via `--model`.

The repo is self-contained: the prompt lists ship under `data/` (regenerate them
with `fetch_datasets.py`). Only the base model is fetched from the Hub (one-off,
~3 GB); after it is cached, runs work fully offline (`HF_HUB_OFFLINE=1`).

## Quick start (fresh Mac)

```bash
bash setup.sh                                  # 1. install everything, run tests, download the model (once, ~3 GB)
. .venv/bin/activate                           # 2. activate (each new terminal)
python abliterate.py                           # 3. run it, ~6 min on Apple silicon; saves the edited model
python heal.py                                 # 4. optional: DPO-heal the edit; ~1 h first run, ~35 min reruns
python chat.py qwen-abliterated-healed --compare   # 5. chat: original vs healed (/q to quit)
bash reset.sh                                  # 6. wipe and start over (add --full to also delete the download)
```

Step 3 writes the edited model to `qwen-abliterated/` (~6 GB) at the end of every run;
step 4 loads it from there. Step 4 is optional: `heal.py` fine-tunes the edited model
with DPO (chosen = original model's answer, rejected = the edited model's answer, on
benign prompts from `data/`), which recovers answer quality the edit cost — LoRA
training, feasible on a Mac; refusals stay at 0 and on-topic answers go 78% → 97%
on this model. It caches its preference pairs to `dpo_pairs.jsonl`, so a second run
skips the generation step. Step 5 chats with the healed model
(`python chat.py --compare` still works against the unhealed one). Add `--no-save`
to step 3 if you don't want the model written to disk. Two warnings you can ignore on
newer setups: `Redirects are currently not supported in MacOs` (torch) and
`You are sending unauthenticated requests to the HF Hub` (public downloads need no token).

Needs only `python3` (macOS Command Line Tools: `xcode-select --install`), ~4 GB free disk
and internet for step 1. Everything else is installed into `.venv` inside this folder.

Faster run (~2 min, single layer tried with real edits):

```bash
python abliterate.py --n-train 64 --n-test 16 --max-new-tokens 48 --max-candidates 3 --top-k 1
```

## What's here

| File | What it is |
|---|---|
| `abliterate.py` | Abliterates `Qwen/Qwen2.5-1.5B-Instruct` and prints before/after refusal rates |
| `heal.py` | Optional DPO healing of the edited model (recovers answer quality) |
| `chat.py` | Chat with the edited model (and the original side by side) |
| `abliteration_core.py` | The maths (difference-of-means, orthogonalisation), unit-tested |
| `data/` | The prompt lists (mlabonne/harmful_behaviors, mlabonne/harmless_alpaca), committed |
| `fetch_datasets.py` | Regenerates `data/` from the Hub (provenance only; not needed to run) |
| `setup.sh`, `reset.sh` | Fresh-machine install / wipe |
| `tests/` | Regression tests (`python -m pytest -q tests`) |

## Safety

The script is sanitised by default: it prints refusal *rates* on held-out prompts and one
harmless question answered before/after. It never prints harmful completions unless you pass
`--show-outputs`. The prompt sets (the public `mlabonne/harmful_behaviors` and
`mlabonne/harmless_alpaca` datasets) are committed under `data/` so the repo is self-contained;
they are standard refusal-direction research prompts, nothing more.

Removing refusal behaviour does not add capability, and usually costs a little general
quality (heal it with DPO/SFT if you care). The `[refused]` / `[answered]` tag in `chat.py`
is a phrase match; glance at the actual text. The edited model's answers can be vague or
wander — "didn't refuse" is not the same as "answered well".

## Notes

- On Apple silicon the script uses fp32 + eager attention: Qwen2.5 overflows in fp16, and
  MPS SDPA returns NaN for left-padded rows. On CUDA it uses bf16 + SDPA. On CPU it runs but
  is several times slower; use the faster variant above.
- Windows: `setup.sh`/`reset.sh` are bash; run the equivalent by hand
  (`python -m venv .venv`, `.venv\Scripts\activate`, `pip install pytest -r requirements.txt`).

## License

MIT — see [LICENSE](LICENSE). The prompt sets under `data/` come from
[mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) and
[mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca); the
method is from [Arditi et al., NeurIPS 2024](https://arxiv.org/abs/2406.11717).
