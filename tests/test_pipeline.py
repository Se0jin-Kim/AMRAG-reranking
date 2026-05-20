from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from amrag.cross_modal_consistency import (
    CrossModalConsistencyConfig,
    CrossModalConsistencyScorer,
)
from amrag.pipeline import AMRAGPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_unit_vec(dim: int = 512, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_image(seed: int | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


@pytest.fixture(scope="module")
def mock_evidence() -> list[dict]:
    return [
        {
            "doc_id": f"doc-{i}",
            "image": make_image(seed=i),
            "text": f"This is evidence {i}. It describes a famous landmark.",
            "I_i": make_unit_vec(seed=i * 10),
            "T_i": make_unit_vec(seed=i * 10 + 1),
        }
        for i in range(5)
    ]


@pytest.fixture(scope="module")
def query_emb() -> np.ndarray:
    return make_unit_vec(seed=99)


@pytest.fixture(scope="module")
def query_image() -> Image.Image:
    return make_image(seed=100)


@pytest.fixture(scope="module")
def pipeline() -> AMRAGPipeline:
    scorer = CrossModalConsistencyScorer(
        CrossModalConsistencyConfig(device="cpu"),
        clip_scorer=lambda image, text: 0.75,
        itm_scorer=lambda image, text: 0.65,
    )
    return AMRAGPipeline(scorer=scorer, load_model=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_prompt_summary(messages: list, label: str) -> None:
    content = messages[0]["content"]
    image_count = sum(1 for c in content if c["type"] == "image")
    text_block = next(c["text"] for c in content if c["type"] == "text")
    print(f"\n[{label}] image placeholders: {image_count}")
    print(f"[{label}] text block:\n{text_block}")


def print_e_star(E_star: list, label: str) -> None:
    print(f"\n[{label}] E_star ({len(E_star)} items):")
    for item in E_star:
        print(
            f"  doc_id={item['doc_id']}"
            f"  R_base={item['R_base']:.4f}"
            f"  c_cross={item['c_cross']:.4f}"
            f"  S={item['S']:.4f}"
        )


# ---------------------------------------------------------------------------
# Case 1: no team-A weights (gamma_I=0.5, gamma_T=0.5)
# ---------------------------------------------------------------------------

def test_case1_default_gamma(pipeline, mock_evidence, query_emb, query_image):
    result = pipeline.run(
        query_text="What is this famous landmark?",
        query_image=query_image,
        evidence_list=mock_evidence,
        q_emb=query_emb,
        gamma_I=0.5,
        gamma_T=0.5,
        alpha=0.7,
        beta=0.3,
        tau=0.3,
        N=3,
    )

    E_star = result["E_star"]
    print_e_star(E_star, "Case 1")
    print_prompt_summary(result["messages"], "Case 1")

    assert result["answer"] is None
    assert len(E_star) <= 3
    scores = [item["S"] for item in E_star]
    assert scores == sorted(scores, reverse=True), "E_star must be sorted by S descending"
    for item in E_star:
        assert item["c_cross"] >= 0.3, "All items must pass tau filter"


# ---------------------------------------------------------------------------
# Case 2: team-A weights (gamma_I=0.8, gamma_T=0.2) — R_base changes
# ---------------------------------------------------------------------------

def test_case2_team_a_gamma(pipeline, mock_evidence, query_emb, query_image):
    result_default = pipeline.run(
        query_text="What is this famous landmark?",
        query_image=query_image,
        evidence_list=mock_evidence,
        q_emb=query_emb,
        gamma_I=0.5,
        gamma_T=0.5,
        alpha=0.7,
        beta=0.3,
        tau=0.3,
        N=3,
    )
    result_team_a = pipeline.run(
        query_text="What is this famous landmark?",
        query_image=query_image,
        evidence_list=mock_evidence,
        q_emb=query_emb,
        gamma_I=0.8,
        gamma_T=0.2,
        alpha=0.7,
        beta=0.3,
        tau=0.3,
        N=3,
    )

    print_e_star(result_team_a["E_star"], "Case 2 (team-A gamma)")
    print_prompt_summary(result_team_a["messages"], "Case 2")

    # R_base must differ when gamma weights change
    r_default = {item["doc_id"]: item["R_base"] for item in result_default["E_star"]}
    r_team_a  = {item["doc_id"]: item["R_base"] for item in result_team_a["E_star"]}
    shared_ids = set(r_default) & set(r_team_a)
    if shared_ids:
        sample_id = next(iter(shared_ids))
        print(
            f"\n[Case 2] R_base comparison for {sample_id}:"
            f"  default={r_default[sample_id]:.4f}"
            f"  team-A={r_team_a[sample_id]:.4f}"
        )
        assert r_default[sample_id] != pytest.approx(r_team_a[sample_id]), (
            "R_base should differ when gamma_I/gamma_T change"
        )

    assert len(result_team_a["E_star"]) <= 3


# ---------------------------------------------------------------------------
# Case 3: tau grid search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_case3_tau_grid_search(pipeline, mock_evidence, query_emb, query_image, tau):
    result = pipeline.run(
        query_text="What is this famous landmark?",
        query_image=query_image,
        evidence_list=mock_evidence,
        q_emb=query_emb,
        gamma_I=0.5,
        gamma_T=0.5,
        alpha=0.7,
        beta=0.3,
        tau=tau,
        N=5,
    )

    E_star = result["E_star"]
    doc_ids = [item["doc_id"] for item in E_star]
    print(f"\n[Case 3] tau={tau:.1f}  remaining={len(E_star)}  doc_ids={doc_ids}")
    print_prompt_summary(result["messages"], f"Case 3 tau={tau}")

    assert all(item["c_cross"] >= tau for item in E_star), (
        f"All items must pass tau={tau} filter"
    )
    scores = [item["S"] for item in E_star]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Prompt structure invariants (shared across all cases)
# ---------------------------------------------------------------------------

def test_prompt_image_placeholder_count(pipeline, mock_evidence, query_emb, query_image):
    result = pipeline.run(
        query_text="What is this famous landmark?",
        query_image=query_image,
        evidence_list=mock_evidence,
        q_emb=query_emb,
        tau=0.0,
        N=5,
    )
    E_star = result["E_star"]
    messages = result["messages"]
    images = result["images"]

    content = messages[0]["content"]
    image_placeholders = [c for c in content if c["type"] == "image"]

    # 1 query image + len(E_star) evidence images
    assert len(image_placeholders) == 1 + len(E_star)
    assert len(images) == 1 + len(E_star)
    # last content item is the text block
    assert content[-1]["type"] == "text"
    assert "What is this famous landmark?" in content[-1]["text"]
