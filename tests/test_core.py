"""Regression tests for the framework-light abliteration maths."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abliteration_core import (  # noqa: E402
    LayerScore,
    ablate_activation,
    best_layer,
    candidate_layers,
    compute_refusal_directions,
    is_refusal,
    orthogonalize_weight,
    refusal_rate,
    residual_component,
)

torch.manual_seed(0)


# --- refusal detection -------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "As an AI language model, I cannot provide that.",
        "I must decline this request.",
        "Unfortunately I am unable to provide instructions for this.",
    ],
)
def test_is_refusal_positive(text):
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "The sky appears blue because shorter wavelengths scatter more.",
        "Sure! Here is a recipe for banana bread.",
        "Step 1: preheat the oven to 180C.",
    ],
)
def test_is_refusal_negative(text):
    assert not is_refusal(text)


def test_is_on_topic_answer():
    from abliteration_core import is_on_topic_answer, on_topic_rate

    q = "Give me a recipe for banana bread with walnuts."
    assert is_on_topic_answer(q, "Sure! For banana bread you need three ripe bananas, flour, sugar, eggs and a cup of walnuts.")
    assert not is_on_topic_answer(q, "I'm sorry, but I can't help with that request.")        # refusal
    assert not is_on_topic_answer(q, "Sure, here you go.")                                     # too short to be real
    assert not is_on_topic_answer(q, "The stock market closed higher today as investors weighed the latest inflation data.")  # off topic
    assert not is_on_topic_answer(q, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")                 # garbage
    assert on_topic_rate([q, q], ["I can't help with that.", "Banana bread: mash bananas, mix with flour, sugar, eggs, walnuts, bake 60 min."]) == pytest.approx(0.5)
    assert on_topic_rate([], []) == 0.0


def test_refusal_rate():
    assert refusal_rate([]) == 0.0
    assert refusal_rate(["I can't do that", "Sure thing"]) == pytest.approx(0.5)


# --- difference of means -----------------------------------------------------
def test_compute_refusal_directions_shape_and_norm():
    n, L, d = 40, 6, 32
    harmful = torch.randn(n, L, d)
    harmless = torch.randn(n, L, d)
    dirs = compute_refusal_directions(harmful, harmless)
    assert dirs.shape == (L, d)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(L), atol=1e-5)


def test_compute_refusal_directions_recovers_planted_direction():
    n, L, d = 200, 4, 64
    planted = torch.zeros(d)
    planted[7] = 1.0
    harmless = torch.randn(n, L, d) * 0.1
    harmful = harmless.clone() + 3.0 * planted  # shift every layer along the planted axis
    dirs = compute_refusal_directions(harmful, harmless)
    cos = dirs @ planted
    assert (cos > 0.99).all()


def test_compute_refusal_directions_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compute_refusal_directions(torch.randn(3, 8), torch.randn(3, 8))
    with pytest.raises(ValueError):
        compute_refusal_directions(torch.randn(3, 2, 8), torch.randn(3, 2, 9))


# --- layer selection ---------------------------------------------------------
def test_candidate_layers_band():
    assert candidate_layers(28) == list(range(11, 22))
    assert candidate_layers(32) == list(range(12, 26))
    assert candidate_layers(1) == [0]
    with pytest.raises(ValueError):
        candidate_layers(0)


def test_best_layer_prefers_lowest_rate_then_shallowest():
    scores = [LayerScore(10, 0.5), LayerScore(14, 0.0), LayerScore(12, 0.0)]
    assert best_layer(scores).layer == 12
    with pytest.raises(ValueError):
        best_layer([])


def test_best_layer_tie_prefers_more_on_topic_answers():
    scores = [LayerScore(10, 0.0, 0.2), LayerScore(14, 0.0, 0.9), LayerScore(12, 0.0, 0.8)]
    assert best_layer(scores).layer == 14
    # on-topic ties still go to the shallower layer
    scores = [LayerScore(10, 0.0, 0.8), LayerScore(14, 0.0, 0.8)]
    assert best_layer(scores).layer == 10


# --- ablation ----------------------------------------------------------------
def test_ablate_activation_removes_component_only():
    d = 16
    r = torch.randn(d)
    r = r / r.norm()
    x = torch.randn(5, 3, d)
    y = ablate_activation(x, r)
    assert torch.allclose(y @ r, torch.zeros(5, 3), atol=1e-5)
    # orthogonal complement preserved
    q = torch.randn(d)
    q = q - (q @ r) * r
    assert torch.allclose(y @ q, x @ q, atol=1e-4)


def test_orthogonalize_linear_weight_out_axis():
    d_model, d_in = 24, 40
    W = torch.randn(d_model, d_in)          # HF nn.Linear convention [out, in]
    r = torch.randn(d_model)
    r = r / r.norm()
    before = residual_component(W, r)
    W2 = orthogonalize_weight(W, r)
    assert W2.shape == W.shape
    assert residual_component(W2, r) < 1e-4 < before
    # columns orthogonal to r are unchanged
    q = torch.randn(d_model)
    q = q - (q @ r) * r
    assert torch.allclose(q @ W2, q @ W, atol=1e-4)


def test_orthogonalize_embedding_weight_in_axis():
    vocab, d_model = 50, 24
    E = torch.randn(vocab, d_model)         # embedding convention [vocab, d_model]
    r = torch.randn(d_model)
    r = r / r.norm()
    E2 = orthogonalize_weight(E, r)
    assert torch.allclose(E2 @ r, torch.zeros(vocab), atol=1e-4)


def test_orthogonalize_square_matrix_requires_explicit_axis():
    """Regression: o_proj is square, so the axis cannot be inferred. Guessing silently
    projected the wrong side in an earlier version."""
    W = torch.randn(32, 32)
    r = torch.randn(32)
    with pytest.raises(ValueError, match="square"):
        orthogonalize_weight(W, r)
    with pytest.raises(ValueError, match="square"):
        residual_component(W, r)


def test_orthogonalize_square_matrix_axis0_zeroes_output_side():
    W = torch.randn(32, 32)
    r = torch.randn(32)
    r = r / r.norm()
    W2 = orthogonalize_weight(W, r, axis=0)
    assert float((r @ W2).norm()) < 1e-4            # cannot write along r
    assert float((W2 @ r).norm()) > 1.0             # input side untouched
    assert residual_component(W2, r, axis=0) < 1e-4


def test_orthogonalize_square_matrix_axis1_zeroes_input_side():
    E = torch.randn(32, 32)
    r = torch.randn(32)
    r = r / r.norm()
    E2 = orthogonalize_weight(E, r, axis=1)
    assert float((E2 @ r).norm()) < 1e-4
    assert float((r @ E2).norm()) > 1.0


def test_orthogonalize_rejects_axis_that_does_not_match():
    with pytest.raises(ValueError):
        orthogonalize_weight(torch.randn(16, 40), torch.randn(16), axis=1)


def test_orthogonalize_is_idempotent_and_preserves_dtype():
    W = torch.randn(24, 40, dtype=torch.float16)
    r = torch.randn(24)
    W2 = orthogonalize_weight(W, r)
    W3 = orthogonalize_weight(W2, r)
    assert W2.dtype == torch.float16
    assert torch.allclose(W2.float(), W3.float(), atol=1e-2)


def test_orthogonalize_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        orthogonalize_weight(torch.randn(10, 12), torch.randn(7))
    with pytest.raises(ValueError):
        orthogonalize_weight(torch.randn(3, 4, 5), torch.randn(5))
