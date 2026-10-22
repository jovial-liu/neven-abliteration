#!/usr/bin/env python3
"""Chat with the abliterated model (and, optionally, the original side by side).

Run abliterate.py once (it saves the edited model to qwen-abliterated/), then:

    python chat.py                       # chat with the edited model
    python chat.py --compare             # original vs edited, same question
    python chat.py some/other/dir        # any saved model or HF model id

Type a question and press Enter. Type /q (or press Ctrl+C) to quit.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.stdout.reconfigure(line_buffering=True)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abliterate import (  # noqa: E402
    DEFAULT_SAVE_DIR, bold, chat_encode, dim, green, pick_device, pick_dtype, red, teal,
)
from abliteration_core import is_refusal  # noqa: E402

ORIGINAL = "Qwen/Qwen2.5-1.5B-Instruct"
QUIT_COMMANDS = ("/q", "/quit", "/exit")


def is_quit(line: str) -> bool:
    return line.strip().lower() in QUIT_COMMANDS


def missing_local_model(path_or_id: str) -> bool:
    """True when the argument is clearly meant as a local folder but it isn't there
    (so we can explain instead of letting transformers try the HF Hub)."""
    p = Path(path_or_id).expanduser()
    if p.is_dir():
        return False
    looks_local = (
        path_or_id.startswith(("./", "../", "/", "~"))
        or path_or_id.endswith(("-abliterated", "-healed"))
    )
    return bool(looks_local)


def load(path_or_id: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn = "eager" if device.type == "mps" else "sdpa"
    tok = AutoTokenizer.from_pretrained(path_or_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=dtype, attn_implementation=attn).to(device).eval()
    return tok, model


@torch.no_grad()
def answer(tok, model, question: str, device, max_new_tokens: int) -> str:
    enc = chat_encode(tok, [question], device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def label(text: str) -> str:
    return red("[refused]") if is_refusal(text) else green("[answered]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=str(DEFAULT_SAVE_DIR),
                    help="saved edited model folder or HF model id (default: qwen-abliterated written by abliterate.py)")
    ap.add_argument("--compare", action="store_true", help=f"also load the original ({ORIGINAL}) and show both answers")
    ap.add_argument("--original", default=ORIGINAL)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    args = ap.parse_args()

    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    if missing_local_model(args.model):
        print(red(f"No saved model at '{args.model}'."))
        cmd = "python abliterate.py" if Path(args.model) == DEFAULT_SAVE_DIR \
            else f"python abliterate.py --save {args.model}"
        print(f"Run abliterate.py first (a few minutes; it saves the edited model), then try again:\n  {bold(cmd)}")
        return 2

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(dim(f"loading {args.model} on {device.type}…"))
    tok_e, model_e = load(args.model, device, dtype)
    models = [("edited", tok_e, model_e)]
    if args.compare:
        print(dim(f"loading {args.original}…"))
        tok_o, model_o = load(args.original, device, dtype)
        models.insert(0, ("original", tok_o, model_o))

    print(bold("\nAsk anything. Type /q to quit.\n"))
    try:
        while True:
            q = input(teal("you> ")).strip()
            if is_quit(q):
                break
            if not q:
                continue
            for name, tok, model in models:
                a = answer(tok, model, q, device, args.max_new_tokens)
                print(f"\n{bold(name)} {label(a)}\n{a}\n")
            print(dim("─" * 60))
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
