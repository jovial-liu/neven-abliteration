#!/usr/bin/env python3
"""Heal the abliterated model with DPO (preference fine-tuning, LoRA).

Abliteration removes refusals but dents answer quality: some answers stop
refusing yet wander off-topic. This script fixes that direction of damage:

  1. Ask the *original* and the *abliterated* model the same benign prompts
     (from data/harmless_alpaca.csv). chosen = original answer,
     rejected = abliterated answer.
  2. Train the abliterated model with DPO on those pairs (LoRA, fp32-friendly
     on a Mac) plus a chosen-answer NLL anchor, so answers are pulled *toward*
     the original model's quality, not just apart from the bad ones.
  3. Verify on held-out prompts that refusals stay at 0 and answers improve,
     then save the merged model to qwen-abliterated-healed/.

LoRA deliberately avoids o_proj and down_proj: those matrices write into the
residual stream and were orthogonalised against the refusal direction — letting
LoRA modify them could relearn writing along it.

Usage:
    python heal.py                    # first run ~1 h on Apple silicon (pair generation dominates);
                                      # reruns ~35 min using the cached pairs
    python heal.py --fresh-pairs      # ignore dpo_pairs.jsonl and regenerate the pairs
    python chat.py qwen-abliterated-healed --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abliterate import (  # noqa: E402
    DATA_DIR,
    DEFAULT_SAVE_DIR,
    _read_prompts,
    banner,
    bold,
    dim,
    generate,
    green,
    pick_device,
    pick_dtype,
    red,
)
from abliteration_core import is_refusal, on_topic_rate, refusal_rate  # noqa: E402
from chat import load as load_model  # noqa: E402

DEFAULT_SAVE = Path(__file__).resolve().parent / "qwen-abliterated-healed"
PAIRS_CACHE = Path(__file__).resolve().parent / "dpo_pairs.jsonl"
ORIGINAL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 640

# Everything that writes into the residual stream (o_proj, down_proj,
# embed_tokens) is orthogonalised against the refusal direction and must stay
# that way; LoRA goes only on matrices that read from it.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_SAVE_DIR, help="abliterated model to heal")
    ap.add_argument("--original", default=ORIGINAL, help="the unedited model (source of chosen answers)")
    ap.add_argument("--save", type=Path, default=DEFAULT_SAVE, help="where to write the healed model")
    ap.add_argument("--prompts", type=int, default=768, help="benign prompts to generate pairs from")
    ap.add_argument("--n-eval", type=int, default=32, help="held-out prompts per class for the report")
    ap.add_argument("--max-new-tokens", type=int, default=112, help="answer length when generating pairs")
    ap.add_argument("--max-len", type=int, default=640, help="training window cap (lower it if memory pressure hits)")
    ap.add_argument("--batch-size", type=int, default=16, help="prompts per generation batch")
    ap.add_argument("--train-batch", type=int, default=2, help="preference pairs per optimiser step")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL trust region)")
    ap.add_argument("--sft-weight", type=float, default=0.5, help="weight of the chosen-answer NLL anchor")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--fresh-pairs", action="store_true", help="regenerate pairs even if the cache exists")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    return ap.parse_args(argv)


def filter_pairs(prompts, chosen, rejected) -> list[dict]:
    """Keep pairs where the original answered well, the edit actually changed
    the answer (identical pairs have zero DPO gradient), and both are
    substantial enough to learn from."""
    pairs = []
    for p, c, r in zip(prompts, chosen, rejected):
        c, r = c.strip(), r.strip()
        if c == r or is_refusal(c) or len(c.split()) < 5 or len(r.split()) < 3:
            continue
        pairs.append({"prompt": p, "chosen": c, "rejected": r})
    return pairs


def encode_example(tok, prompt: str, answer: str, max_len: int = MAX_LEN):
    """Token ids and a loss mask that marks only the answer tokens."""
    p_text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                     tokenize=False, add_generation_prompt=True)
    p_len = len(tok(p_text)["input_ids"])
    ids = tok(p_text + answer)["input_ids"]
    if len(ids) > max_len:
        ids = ids[:max_len]
    mask = [0] * min(p_len, len(ids)) + [1] * (len(ids) - min(p_len, len(ids)))
    return ids, mask


def collate(rows, pad_id, device):
    maxlen = max(len(ids) for ids, _ in rows)
    input_ids, attn, loss = [], [], []
    for ids, mask in rows:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        loss.append(mask + [0] * pad)
    to = lambda x: torch.tensor(x, device=device)  # noqa: E731
    return to(input_ids), to(attn), to(loss).bool()


def seq_logprobs(model, input_ids, attn, loss_mask) -> torch.Tensor:
    """Sum of answer-token log-probs, one scalar per sequence."""
    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok * loss_mask[:, 1:]).sum(-1)


def dpo_loss_from_logprobs(pol_c, pol_r, ref_c, ref_r, beta: float):
    margins = (pol_c - ref_c) - (pol_r - ref_r)
    loss = -F.logsigmoid(beta * margins).mean()
    return loss, margins


def attach_lora(model, r: int):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def evaluate(model, tok, harmful, benign, device, max_new_tokens, batch_size):
    h_out = generate(model, tok, harmful, device, max_new_tokens, batch_size)
    b_out = generate(model, tok, benign, device, max_new_tokens, batch_size)
    return {
        "refusal": refusal_rate(h_out),
        "on_topic": on_topic_rate(harmful, h_out),
        "benign": on_topic_rate(benign, b_out),
    }


def report(label, stats):
    ref = red(f"{stats['refusal']:.0%}")
    ont = bold(f"{stats['on_topic']:.0%}")
    ben = bold(f"{stats['benign']:.0%}")
    print(f"{label:<13} refusals {ref}   on-topic {ont}   benign quality {ben}")


def run(args) -> None:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    t_start = time.time()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    harmful_eval = _read_prompts(DATA_DIR / "harmful_behaviors_test.csv")[: args.n_eval]
    alpaca = _read_prompts(DATA_DIR / "harmless_alpaca.csv")
    # Prompt slices are kept disjoint from the first 256 used for the direction.
    benign_train = alpaca[512 : 512 + args.prompts]
    benign_eval = alpaca[4000 : 4000 + args.n_eval]

    tok, abl = load_model(str(args.model), device, dtype)

    # ---- 1. pairs -----------------------------------------------------------
    if PAIRS_CACHE.exists() and not args.fresh_pairs:
        with open(PAIRS_CACHE, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        banner("1 · Preference pairs", f"reusing {len(pairs)} cached pairs from {PAIRS_CACHE.name}")
    else:
        banner("1 · Preference pairs", "chosen = original answer, rejected = abliterated answer, on benign prompts")
        print(dim(f"also loading the original ({args.original})…"))
        _, orig = load_model(args.original, device, dtype)
        t0 = time.time()
        chosen = generate(orig, tok, benign_train, device, args.max_new_tokens, args.batch_size)
        rejected = generate(abl, tok, benign_train, device, args.max_new_tokens, args.batch_size)
        print(dim(f"both models answered {len(benign_train)} benign prompts  ({time.time() - t0:.0f}s)"))
        del orig
        pairs = filter_pairs(benign_train, chosen, rejected)
        dropped = len(benign_train) - len(pairs)
        print(f"{len(pairs)} usable pairs  {dim(f'({dropped} dropped: identical, refusal, or too short)')}")
        with open(PAIRS_CACHE, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(dim(f"cached to {PAIRS_CACHE.name}"))

    # ---- 2. dataset ---------------------------------------------------------
    banner("2 · Tokenise", f"{len(pairs)} pairs, windows up to {args.max_len} tokens")
    pad_id = tok.pad_token_id or tok.eos_token_id
    rows = []   # per pair: ((chosen ids, mask), (rejected ids, mask))
    skipped = 0
    for pair in pairs:
        try:
            c = encode_example(tok, pair["prompt"], pair["chosen"], args.max_len)
            r = encode_example(tok, pair["prompt"], pair["rejected"], args.max_len)
        except Exception:
            skipped += 1
            continue
        if not any(c[1]) or not any(r[1]):
            skipped += 1        # the prompt alone fills the window
            continue
        rows.append((c, r))
    if skipped:
        print(dim(f"skipped {skipped} over-long examples"))

    # ---- 3. before ----------------------------------------------------------
    banner("3 · Before healing", "the abliterated model: no refusals, but answers wander")
    t0 = time.time()
    before = evaluate(abl, tok, harmful_eval, benign_eval, device, 64, args.batch_size)
    report("abliterated", before)
    print(dim(f"({time.time() - t0:.0f}s)"))

    # ---- 4. train -----------------------------------------------------------
    banner("4 · DPO", f"LoRA r={args.lora_r} on {', '.join(LORA_TARGETS)}; "
                      f"β={args.beta}, lr={args.lr}, {args.epochs} epoch(s), NLL anchor {args.sft_weight}")
    torch.manual_seed(0)
    peft_model = attach_lora(abl, args.lora_r)

    # Reference log-probs are the base model's (LoRA disabled); precompute once.
    refs = torch.zeros(len(rows), 2)
    t0 = time.time()
    with torch.no_grad(), peft_model.disable_adapter():
        for i in range(0, len(rows), 2):
            chunk = rows[i : i + 2]
            for j, side in enumerate((0, 1)):
                batch = collate([r[side] for r in chunk], pad_id, device)
                refs[i : i + len(chunk), j] = seq_logprobs(peft_model, *batch).cpu()
    print(dim(f"reference log-probs precomputed  ({time.time() - t0:.0f}s)"))

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    n_pairs = len(rows)
    steps_per_epoch = (n_pairs + args.train_batch - 1) // args.train_batch
    total_steps = steps_per_epoch * args.epochs
    order = torch.randperm(n_pairs).tolist()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if epoch > 0:
            order = torch.randperm(n_pairs).tolist()
        for s in range(0, n_pairs, args.train_batch):
            idx = order[s : s + args.train_batch]
            batch_rows = [rows[i] for i in idx]
            c = collate([r[0] for r in batch_rows], pad_id, device)
            r = collate([r[1] for r in batch_rows], pad_id, device)
            pol_c = seq_logprobs(peft_model, *c)
            pol_r = seq_logprobs(peft_model, *r)
            ref = refs[idx].to(device)
            dpo, margins = dpo_loss_from_logprobs(pol_c, pol_r, ref[:, 0], ref[:, 1], args.beta)
            n_tokens = c[2][:, 1:].sum().clamp_min(1)
            loss = dpo + args.sft_weight * -(pol_c.sum() / n_tokens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % 25 == 0 or step == total_steps:
                acc = (margins > 0).float().mean().item()
                print(f"  step {step:>4}/{total_steps}  dpo {dpo.item():.4f}  "
                      f"margin {margins.mean().item():+.3f}  pairs-beating-ref {acc:.0%}  "
                      f"{dim(f'({time.time() - t0:.0f}s)')}")

    healed = peft_model.merge_and_unload()

    # ---- 5. after -----------------------------------------------------------
    banner("5 · After healing", "refusals must stay at 0; on-topic answers should rise")
    after = evaluate(healed, tok, harmful_eval, benign_eval, device, 64, args.batch_size)
    report("healed", after)

    banner("Done")
    b_ref, a_ref = red(f"{before['refusal']:.0%}"), green(bold(f"{after['refusal']:.0%}"))
    a_ont = green(bold(f"{after['on_topic']:.0%}"))
    a_ben = bold(f"{after['benign']:.0%}")
    print(f"refusals   : {b_ref} -> {a_ref}   {dim('(must stay 0)')}")
    print(f"on-topic   : {before['on_topic']:.0%} -> {a_ont}   {dim('(the wandering share this fixes)')}")
    print(f"benign     : {before['benign']:.0%} -> {a_ben}")
    print(f"total time : {time.time() - t_start:.0f}s")

    args.save.mkdir(parents=True, exist_ok=True)
    healed.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    print(f"\nsaved the healed model to {args.save}")
    print(dim("chat with it:   python chat.py qwen-abliterated-healed --compare"))


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print(red("\nstopped by user"))
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
