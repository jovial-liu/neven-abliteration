"""Unit tests for the DPO healing stage (no model download, no training)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import heal  # noqa: E402


def test_parse_args_defaults():
    args = heal.parse_args([])
    assert args.model.name == "qwen-abliterated"
    assert args.save.name == "qwen-abliterated-healed"
    assert args.beta == 0.1 and args.sft_weight == 0.5 and args.lr == 1e-5
    assert args.epochs == 1 and args.lora_r == 16
    assert not args.fresh_pairs


def test_lora_targets_never_touch_residual_stream_writers():
    """o_proj / down_proj / embed_tokens write into the residual stream and are
    orthogonalised against the refusal direction; LoRA must not modify them."""
    assert set(heal.LORA_TARGETS) == {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
    assert not set(heal.LORA_TARGETS) & {"o_proj", "down_proj", "embed_tokens"}


def test_filter_pairs_drops_identical_refusing_and_thin_answers():
    prompts = ["p1", "p2", "p3", "p4", "p5"]
    chosen = [
        "Sure, here is a full answer with several words.",   # good
        "I'm sorry, I can't help with that.",                # refusal -> drop
        "ok",                                                 # too short -> drop
        "The same answer text",                               # identical -> drop
        "Another decent answer with words here.",
    ]
    rejected = [
        "wandering text with words",
        "some text",
        "short but three words yes",
        "The same answer text",
        "drifted off somewhere else entirely",
    ]
    pairs = heal.filter_pairs(prompts, chosen, rejected)
    assert [p["prompt"] for p in pairs] == ["p1", "p5"]
    assert pairs[0]["chosen"].startswith("Sure")
    assert pairs[0]["rejected"] == "wandering text with words"


class _MockLM(torch.nn.Module):
    """Returns logits that hotly favour the *current* token, so the expected
    log-prob of any (position -> next token) pair can be computed by hand."""

    def __init__(self, vocab=5):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, attention_mask):
        logits = torch.nn.functional.one_hot(input_ids, self.vocab).float() * 3.0
        return SimpleNamespace(logits=logits)


def test_seq_logprobs_shifts_and_masks_correctly():
    # ids [2, 0, 3]; only the last token (3) is an answer token, predicted from
    # position 1 whose logits favour token 0 with logit 3, all others logit 0.
    ids = torch.tensor([[2, 0, 3]])
    attn = torch.ones_like(ids)
    mask = torch.tensor([[False, False, True]])
    got = heal.seq_logprobs(_MockLM(), ids, attn, mask)
    logp = torch.log_softmax(torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]), dim=-1)[3]
    assert torch.allclose(got, logp.unsqueeze(0), atol=1e-6)

    # no answer tokens -> log-prob 0, and prompt-only positions never contribute
    assert float(heal.seq_logprobs(_MockLM(), ids, attn, torch.zeros_like(mask).bool())) == 0.0


def test_seq_logprobs_matches_a_reference_implementation_on_random_data():
    torch.manual_seed(0)
    b, t, vocab = 3, 12, 40
    model = _MockLM(vocab)
    ids = torch.randint(0, vocab, (b, t))
    attn = torch.ones_like(ids)
    mask = torch.rand(b, t) < 0.5
    mask[:, :4] = False
    got = heal.seq_logprobs(model, ids, attn, mask)
    logits = torch.nn.functional.one_hot(ids, vocab).float() * 3.0
    ref = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    ref = ref.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    ref = (ref * mask[:, 1:]).sum(-1)
    assert torch.allclose(got, ref, atol=1e-5)


def test_dpo_loss_matches_the_formula():
    pol_c = torch.tensor([1.0, 0.2])
    pol_r = torch.tensor([0.2, -0.1])
    ref_c = torch.tensor([0.5, 0.0])
    ref_r = torch.tensor([0.4, 0.3])
    beta = 0.1
    loss, margins = heal.dpo_loss_from_logprobs(pol_c, pol_r, ref_c, ref_r, beta)
    m = torch.tensor([(1.0 - 0.5) - (0.2 - 0.4), (0.2 - 0.0) - (-0.1 - 0.3)])
    assert torch.allclose(margins, m)
    expected = -sum(math.log(torch.sigmoid(beta * x).item()) for x in m) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-5)

    # perfectly ordered pairs with margin 0 -> loss = log 2
    z = torch.zeros(1)
    loss0, _ = heal.dpo_loss_from_logprobs(z, z, z, z, beta)
    assert loss0.item() == pytest.approx(math.log(2.0), abs=1e-6)


def test_encode_example_masks_only_answer_tokens():
    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<prompt>"

        def __call__(self, text, **kw):
            ids = [10] * 4 + ([11] * len(text) if "<prompt>" not in text else [])
            # prompt text -> 4 ids; prompt+answer -> 4 + len(answer) ids
            n_answer = 0 if "<prompt>" in text and text == "<prompt>" else len(text) - len("<prompt>")
            ids = [10] * 4 + [11] * n_answer
            return {"input_ids": ids}

    ids, mask = heal.encode_example(_Tok(), "question", "ans")
    assert len(ids) == len(mask) == 7
    assert mask == [0, 0, 0, 0, 1, 1, 1]
