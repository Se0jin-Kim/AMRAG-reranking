from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from amrag.cross_modal_consistency import (
    CrossModalConsistencyConfig,
    CrossModalConsistencyScorer,
    CrossModalInput,
    MissingModalityError,
)
from amrag.cross_modal_consistency import scoring as scoring_module
from amrag.cross_modal_consistency.model_loader import ModelBundle, load_model_bundle
from amrag.cross_modal_consistency.model_loader import resolve_device


def make_scorer(clip_score=0.8, itm_score=0.6, alpha=0.5, beta=0.5, gamma=0.0):
    return CrossModalConsistencyScorer(
        CrossModalConsistencyConfig(alpha=alpha, beta=beta, gamma=gamma, device="cpu"),
        clip_scorer=lambda image, text: clip_score,
        itm_scorer=lambda image, text: itm_score,
    )


@pytest.fixture
def valid_image():
    return Image.new("RGB", (8, 8), color="white")


def test_valid_image_text_pair_returns_structured_score(valid_image):
    scorer = make_scorer(clip_score=0.8, itm_score=0.6)

    result = scorer.score_document(
        query="What animal is shown?",
        d_i={"id": "doc-1"},
        image_i=valid_image,
        text_i="A cat sitting on a mat.",
        w_text=0.7,
        w_image=0.3,
    )

    assert result["doc_id"] == "doc-1"
    assert result["c_clip"] == pytest.approx(0.8)
    assert result["c_itm"] == pytest.approx(0.6)
    assert result["c_cross"] == pytest.approx(0.7)
    assert result["metadata"]["w_text_w_image_used"] is False
    assert result["metadata"]["normalization_method"]["itm"] == "injected_scorer"
    assert result["metadata"]["model_names"]["clip"] == "openai/clip-vit-base-patch32"


def test_mismatched_pair_scores_lower_when_component_scores_are_lower():
    matched = make_scorer(clip_score=0.9, itm_score=0.9)
    mismatched = make_scorer(clip_score=0.2, itm_score=0.1)

    matched_result = matched.score_document(
        query=None,
        d_i={"id": "matched"},
        image_i=object(),
        text_i="A red car.",
    )
    mismatched_result = mismatched.score_document(
        query=None,
        d_i={"id": "mismatched"},
        image_i=object(),
        text_i="A bowl of soup.",
    )

    assert mismatched_result["c_cross"] < matched_result["c_cross"]


def test_missing_image_raises_explicit_error():
    scorer = make_scorer()

    with pytest.raises(MissingModalityError, match="image_i is required"):
        scorer.score_document(
            query=None,
            d_i={"id": "doc-1"},
            image_i=None,
            text_i="Text is present.",
        )


def test_missing_text_raises_explicit_error():
    scorer = make_scorer()

    with pytest.raises(MissingModalityError, match="text_i is required"):
        scorer.score_document(
            query=None,
            d_i={"id": "doc-1"},
            image_i=object(),
            text_i=" ",
        )


def test_batch_scoring_preserves_order_and_outputs():
    scorer = make_scorer(clip_score=0.4, itm_score=0.8)
    inputs = [
        CrossModalInput(None, {"id": "doc-1"}, object(), "First text."),
        CrossModalInput(None, {"id": "doc-2"}, object(), "Second text."),
    ]

    results = scorer.score_batch(inputs)

    assert [result["doc_id"] for result in results] == ["doc-1", "doc-2"]
    assert [result["c_cross"] for result in results] == [
        pytest.approx(0.6),
        pytest.approx(0.6),
    ]


@pytest.mark.parametrize(
    ("clip_score", "itm_score"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ],
)
def test_score_range_validity(clip_score, itm_score):
    scorer = make_scorer(clip_score=clip_score, itm_score=itm_score)

    result = scorer.score_document(
        query=None,
        d_i={"id": "doc-1"},
        image_i=object(),
        text_i="A valid caption.",
    )

    assert 0.0 <= result["c_clip"] <= 1.0
    assert 0.0 <= result["c_itm"] <= 1.0
    assert 0.0 <= result["c_cross"] <= 1.0


def test_alpha_beta_configurability_changes_cross_score():
    scorer = make_scorer(clip_score=0.2, itm_score=0.8, alpha=0.25, beta=0.75)

    result = scorer.score_document(
        query=None,
        d_i={"id": "doc-1"},
        image_i=object(),
        text_i="A valid caption.",
    )

    assert result["c_cross"] == pytest.approx(0.65)


def test_alpha_beta_must_sum_to_one_by_default():
    with pytest.raises(ValueError, match="alpha \\+ beta \\+ gamma must equal 1\\.0"):
        CrossModalConsistencyScorer(
            CrossModalConsistencyConfig(alpha=0.3, beta=0.3),
            clip_scorer=lambda image, text: 0.5,
            itm_scorer=lambda image, text: 0.5,
        )


def test_w_text_and_w_image_do_not_affect_cross_score():
    scorer = make_scorer(clip_score=0.7, itm_score=0.5)
    common = {
        "query": "same query",
        "d_i": {"id": "doc-1"},
        "image_i": object(),
        "text_i": "same text",
    }

    first = scorer.score_document(**common, w_text=1.0, w_image=0.0)
    second = scorer.score_document(**common, w_text=0.0, w_image=1.0)

    assert first["c_cross"] == second["c_cross"]


def test_real_itm_softmax_path_reports_metadata():
    class FakeProcessor:
        def __call__(self, **kwargs):
            return {}

    class FakeBlipModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(itm_score=torch.tensor([[0.0, 2.0]]))

    scorer = CrossModalConsistencyScorer(
        CrossModalConsistencyConfig(device="cpu"),
        model_bundle=ModelBundle(
            clip_model=None,
            clip_processor=None,
            blip_model=FakeBlipModel(),
            blip_processor=FakeProcessor(),
            device="cpu",
        ),
        clip_scorer=lambda image, text: 0.5,
    )

    result = scorer.score_document(
        query=None,
        d_i={"id": "doc-1"},
        image_i=object(),
        text_i="A valid caption.",
    )

    assert result["c_itm"] == pytest.approx(float(torch.softmax(torch.tensor([0.0, 2.0]), dim=-1)[1]))
    assert result["metadata"]["normalization_method"]["itm"] == "softmax_matched_probability"


def test_real_itm_1d_two_logit_path_reports_softmax_metadata():
    class FakeProcessor:
        def __call__(self, **kwargs):
            return {}

    class FakeBlipModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(itm_score=torch.tensor([0.0, 2.0]))

    scorer = CrossModalConsistencyScorer(
        CrossModalConsistencyConfig(device="cpu"),
        model_bundle=ModelBundle(
            clip_model=None,
            clip_processor=None,
            blip_model=FakeBlipModel(),
            blip_processor=FakeProcessor(),
            device="cpu",
        ),
        clip_scorer=lambda image, text: 0.5,
    )

    result = scorer.score_document(
        query=None,
        d_i={"id": "doc-1"},
        image_i=object(),
        text_i="A valid caption.",
    )

    expected = float(torch.softmax(torch.tensor([0.0, 2.0]), dim=-1)[1])
    assert result["c_itm"] == pytest.approx(expected)
    assert result["metadata"]["normalization_method"]["itm"] == "softmax_matched_probability"


def test_model_bundle_loads_once_and_is_reused_per_document(monkeypatch):
    class FakeProcessor:
        def __call__(self, **kwargs):
            return {}

    class FakeClipModel:
        def get_image_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0]])

        def get_text_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0]])

    class FakeBlipModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(itm_score=torch.tensor([[0.0, 2.0]]))

    load_calls = []

    def fake_load_model_bundle(config):
        load_calls.append(config)
        return ModelBundle(
            clip_model=FakeClipModel(),
            clip_processor=FakeProcessor(),
            blip_model=FakeBlipModel(),
            blip_processor=FakeProcessor(),
            device="cpu",
        )

    monkeypatch.setattr(scoring_module, "load_model_bundle", fake_load_model_bundle)
    scorer = CrossModalConsistencyScorer(CrossModalConsistencyConfig(device="cpu"))

    scorer.score_document(query=None, d_i={"id": "doc-1"}, image_i=object(), text_i="Text one.")
    scorer.score_document(query=None, d_i={"id": "doc-2"}, image_i=object(), text_i="Text two.")

    assert len(load_calls) == 1


def test_resolve_device_auto_and_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto") == "cuda"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_model_loading_does_not_disable_global_grad(monkeypatch):
    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeProcessor,
        BlipForImageTextRetrieval=FakeModel,
        CLIPModel=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    torch.set_grad_enabled(True)

    try:
        load_model_bundle(CrossModalConsistencyConfig(device="cpu"))
        assert torch.is_grad_enabled() is True
    finally:
        torch.set_grad_enabled(True)
