from __future__ import annotations

import logging
from typing import Any

from amrag.cross_modal_consistency import CrossModalConsistencyScorer

logger = logging.getLogger(__name__)


def get_adaptive_weights(
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma_I: float = 0.5,
    gamma_T: float = 0.5,
    use_adaptive: bool = False,
) -> tuple[float, float]:
    """Return (alpha, beta) for the final reranking score S.

    use_adaptive=False (default):
        Validate and return the fixed alpha / beta hyperparameters.

    use_adaptive=True:
        Derive alpha and beta from gamma_I / gamma_T so that an
        image-heavy query raises the c_cross contribution:
            alpha(q) = 1 - gamma_I  (R_base weight)
            beta(q)  = gamma_I      (c_cross weight)
        The fixed alpha / beta arguments are ignored in this mode.
    """
    if use_adaptive:
        alpha = 1.0 - gamma_I
        beta = gamma_I
    else:
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha and beta must be non-negative.")
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta must equal 1.0, got alpha={alpha}, beta={beta}."
            )
    return alpha, beta


def compute_eta(
    gamma_I: float = 0.5,
    gamma_T: float = 0.5,  # noqa: ARG001  kept for a symmetric call signature
) -> float:
    """Compute query-adaptive consistency scaling factor η(q).

    η is high when the query is image-heavy (gamma_I → 1), meaning
    cross-modal consistency matters more in the final ranking.
    η is low when the query is text-heavy (gamma_I → 0).

        η(q) = gamma_I
    """
    return float(gamma_I)


def _dot(a: Any, b: Any) -> float:
    """Dot product compatible with numpy arrays, torch tensors, and sequences."""
    try:
        return float(a @ b)
    except TypeError:
        if hasattr(a, "__len__") and hasattr(b, "__len__") and len(a) != len(b):
            raise ValueError(
                f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
            )
        return float(sum(x * y for x, y in zip(a, b)))


def compute_R_base(
    q_emb: Any,
    I_i: Any,
    T_i: Any,
    gamma_I: float | None = 0.5,
    gamma_T: float | None = 0.5,
) -> float:
    """Compute retrieval-stage relevance score.

    R_base = gamma_I * R_I + gamma_T * R_T

    R_I and R_T are dot products between the query embedding and the
    image / text embeddings respectively. Assumes pre-normalised embeddings
    (dot product equals cosine similarity).

    Args:
        q_emb: Query embedding vector.
        I_i:   Image embedding for document i.
        T_i:   Text embedding for document i.
        gamma_I: Weight for image relevance. Defaults to 0.5.
        gamma_T: Weight for text relevance. Defaults to 0.5.
    """
    if gamma_I is None:
        gamma_I = 0.5
    if gamma_T is None:
        gamma_T = 0.5

    total = gamma_I + gamma_T
    if total > 1.0 + 1e-6:
        logger.warning(
            "compute_R_base: gamma_I + gamma_T = %.4f > 1.0 — R_base may exceed expected range.",
            total,
        )

    R_I = _dot(q_emb, I_i)
    R_T = _dot(q_emb, T_i)
    return gamma_I * R_I + gamma_T * R_T


def filter_by_consistency(
    evidence_list: list[dict[str, Any]],
    tau: float,
) -> list[dict[str, Any]]:
    """Remove evidence whose c_cross falls below threshold tau."""
    kept = [e for e in evidence_list if e["c_cross"] >= tau]
    removed = len(evidence_list) - len(kept)
    if removed:
        logger.info(
            "filter_by_consistency: removed %d/%d evidence items (tau=%.4f)",
            removed,
            len(evidence_list),
            tau,
        )
    return kept


def compute_reranking_score(
    R_base: float,
    c_clip_scaled: float,
    alpha: float,
    beta: float,
) -> float:
    """Compute final reranking score S = alpha * R_base + beta * c_clip_scaled.

    c_clip_scaled is η(q) · C_clip_norm(Ei) when use_adaptive=True,
    or C_clip_norm(Ei) when use_adaptive=False (η = 1.0).
    """
    return alpha * R_base + beta * c_clip_scaled


def rerank(
    evidence_list: list[dict[str, Any]],
    scorer: CrossModalConsistencyScorer,
    q_emb: Any,
    alpha: float = 0.5,        # noqa: ARG001  kept for backward compatibility
    beta: float = 0.5,         # noqa: ARG001  kept for backward compatibility
    gamma_I: float | None = 0.5,
    gamma_T: float | None = 0.5,
    tau: float = 0.5,
    N: int = 5,
    use_adaptive: bool = False,  # noqa: ARG001  kept for backward compatibility
    lambda_c: float = 0.5,
    query_text: str | None = None,
) -> list[dict[str, Any]]:
    """Score, filter, and rerank evidence documents by cross-modal consistency.

    Scoring formula:
        S_final(q, di) = w_image(q)·S_image(q, di)
                       + w_text(q)·S_text(q, di)
                       + λ·C_cross(di)

        S_image(q, di) = sim(E_text(q), E_image(image_i)) = dot(q_emb, I_i)
        S_text(q, di)  = sim(E_text(q), E_text(text_i))  = dot(q_emb, T_i)
        w_image(q)     = gamma_I
        w_text(q)      = gamma_T
        λ              = lambda_c
        C_cross(di)    = c_cross (CLIP + BLIP ITM blend)

    Args:
        evidence_list: Each item must contain keys
            ``doc_id``, ``image``, ``text``, ``I_i``, and ``T_i``.
        scorer: A ready-to-use CrossModalConsistencyScorer instance.
        q_emb: Query embedding used to compute S_image and S_text.
        alpha: Unused in S formula; kept for backward compatibility.
        beta: Unused in S formula; kept for backward compatibility.
        gamma_I: Image modality weight (w_image). Defaults to 0.5.
        gamma_T: Text modality weight (w_text). Defaults to 0.5.
        tau: Minimum c_cross threshold; items below are discarded before scoring.
        N: Maximum number of results to return.
        use_adaptive: Unused in S formula; kept for backward compatibility.
        lambda_c: Weight for the C_cross consistency term. Defaults to 0.5.
        query_text: Original query string forwarded to the scorer so that
            C_entity (Jaccard entity overlap) can be computed. When None,
            C_entity is set to 0.0 inside the scorer.

    Returns:
        Up to N items sorted by descending S, each containing
        ``doc_id``, ``image``, ``text``, ``R_base``, ``c_cross``, and ``S``.
    """
    gamma_I = gamma_I if gamma_I is not None else 0.5
    gamma_T = gamma_T if gamma_T is not None else 0.5

    # Step 1: compute retrieval scores and consistency scores for every evidence item
    scored: list[dict[str, Any]] = []
    for ev in evidence_list:
        score_result = scorer.score_document(
            query=query_text,
            d_i={"id": ev["doc_id"]},
            image_i=ev["image"],
            text_i=ev["text"],
            doc_id=ev["doc_id"],
        )
        S_image = _dot(q_emb, ev["I_i"])  # sim(E_text(q), E_image(image_i))
        S_text = _dot(q_emb, ev["T_i"])   # sim(E_text(q), E_text(text_i))
        R_base = gamma_I * S_image + gamma_T * S_text  # kept for output
        scored.append({
            "doc_id": ev["doc_id"],
            "image": ev["image"],
            "text": ev["text"],
            "R_base": R_base,
            "c_cross": score_result["c_cross"],
            "_S_image": S_image,
            "_S_text": S_text,
        })

    # Step 2: filter by raw c_cross (tau gate is query-type-independent)
    scored = filter_by_consistency(scored, tau)
    if not scored:
        logger.warning(
            "rerank: all %d evidence items filtered out by tau=%.4f — returning empty list.",
            len(evidence_list),
            tau,
        )
        return []

    # Step 3: compute final score S using new formula
    for item in scored:
        S_image = item.pop("_S_image")
        S_text = item.pop("_S_text")
        item["S"] = (
            gamma_I * S_image
            + gamma_T * S_text
            + lambda_c * item["c_cross"]
        )

    # Step 4: sort descending by S, return top N
    scored.sort(key=lambda x: x["S"], reverse=True)
    return scored[:N]
