"""Regression tests for abliterate.py's device/dtype decisions and hook maths.

These cover the two bugs found during development: fp16 overflow on Qwen2.5 and
NaNs from MPS SDPA with left padding. They do not download a model.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import abliterate  # noqa: E402


@pytest.mark.parametrize(
    "name, device, expected",
    [
        ("auto", "cuda", torch.bfloat16),
        ("auto", "mps", torch.float32),
        ("auto", "cpu", torch.float32),
        ("bfloat16", "mps", torch.bfloat16),
        ("float16", "cuda", torch.float16),
    ],
)
def test_pick_dtype(name, device, expected):
    assert abliterate.pick_dtype(name, torch.device(device)) == expected


def test_chat_label_marks_refusals():
    import chat

    assert "refused" in chat.label("I'm sorry, but I can't help with that.")
    assert "answered" in chat.label("Leaves whisper to ground, crisp air fills the evening.")


def test_chat_quits_on_slash_q_only():
    import chat

    for line in ("/q", " /Q ", "/quit", "/exit"):
        assert chat.is_quit(line), line
    for line in ("", "q", "quit", "how do I /q something", "/qq"):
        assert not chat.is_quit(line), line


def test_chat_explains_missing_saved_model_instead_of_hitting_the_hub(tmp_path, monkeypatch):
    import chat

    monkeypatch.chdir(tmp_path)
    assert chat.missing_local_model("./anything")
    assert chat.missing_local_model("my-qwen-abliterated")
    # real HF ids must still pass through to the Hub
    assert not chat.missing_local_model("Qwen/Qwen2.5-1.5B-Instruct")
    # and an existing folder is fine
    (tmp_path / "qwen-abliterated").mkdir()
    assert not chat.missing_local_model("qwen-abliterated")

    monkeypatch.setattr(sys, "argv", ["chat.py", "missing-abliterated"])
    assert chat.main() == 2  # clean exit, no traceback, no network


def test_saves_by_default_where_chat_looks(monkeypatch, tmp_path):
    """One command runs abliterate.py, the next chats: both must agree on the folder."""
    import chat

    args = abliterate.parse_args([])
    assert not args.no_save and args.save == abliterate.DEFAULT_SAVE_DIR
    assert abliterate.DEFAULT_SAVE_DIR.name == "qwen-abliterated"
    assert abliterate.parse_args(["--no-save"]).no_save
    assert abliterate.parse_args(["--save", "x"]).save == Path("x")

    # chat.py with no argument points at the same folder; missing -> clean exit 2
    monkeypatch.setattr(abliterate, "DEFAULT_SAVE_DIR", tmp_path / "qwen-abliterated")
    monkeypatch.setattr(chat, "DEFAULT_SAVE_DIR", tmp_path / "qwen-abliterated")
    monkeypatch.setattr(sys, "argv", ["chat.py"])
    assert chat.main() == 2


def test_auto_dtype_never_fp16():
    for dev in ("cuda", "mps", "cpu"):
        assert abliterate.pick_dtype("auto", torch.device(dev)) != torch.float16


def test_alpha_sweeps_by_default_and_can_be_pinned():
    assert abliterate.parse_args([]).alpha is None          # sweep
    assert abliterate.parse_args(["--alpha", "0.8"]).alpha == 0.8


class _Block(torch.nn.Module):
    def forward(self, x):
        return (x + 1.0,)  # HF decoder layers return a tuple


class _Inner(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _Model(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.model = _Inner(n)

    def forward(self, x):
        return self.model(x)


class _FakeAttn(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.o_proj = torch.nn.Linear(d, d, bias=False)          # square, like Qwen2.5-1.5B


class _FakeMLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.down_proj = torch.nn.Linear(4 * d, d, bias=False)   # [d_model, d_ff]


class _FakeLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _FakeAttn(d)
        self.mlp = _FakeMLP(d)


class _FakeInner(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(50, d)
        self.layers = torch.nn.ModuleList([_FakeLayer(d) for _ in range(n)])


class _FakeHF(torch.nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.model = _FakeInner(d, n)


def test_orthogonalize_model_blocks_every_writer_including_square_o_proj():
    """End-to-end on a fake HF-shaped model: after the edit, no component that writes
    into the residual stream can produce any output along the refusal direction."""
    torch.manual_seed(0)
    d = 16
    m = _FakeHF(d=d, n=3)
    r = torch.randn(d)
    r = r / r.norm()

    # before: everything writes along r
    assert float((m.model.embed_tokens.weight @ r).norm()) > 0.5
    assert all(float((r @ l.self_attn.o_proj.weight).norm()) > 0.5 for l in m.model.layers)

    abliterate.orthogonalize_model(m, r)

    assert float((m.model.embed_tokens.weight @ r).norm()) < 1e-4
    for l in m.model.layers:
        x = torch.randn(5, d)
        assert torch.allclose(l.self_attn.o_proj(x) @ r, torch.zeros(5), atol=1e-4)
        h = torch.randn(5, 4 * d)
        assert torch.allclose(l.mlp.down_proj(h) @ r, torch.zeros(5), atol=1e-4)


def test_inference_time_ablation_hook_removes_direction_and_is_removed_after():
    d = 8
    r = torch.zeros(d)
    r[2] = 1.0
    m = _Model(3)
    x = torch.randn(4, d)

    with abliterate.inference_time_ablation(m, r):
        y = m(x)
    assert torch.allclose(y @ r, torch.zeros(4), atol=1e-6)      # ablated inside the context

    y2 = m(x)
    assert not torch.allclose(y2 @ r, torch.zeros(4), atol=1e-6)  # hooks removed afterwards
    assert all(len(l._forward_hooks) == 0 for l in m.model.layers)
