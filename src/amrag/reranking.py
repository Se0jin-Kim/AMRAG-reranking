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
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma_I: float | None = 0.5,
    gamma_T: float | None = 0.5,
    tau: float = 0.5,
    N: int = 5,
    use_adaptive: bool = False,
) -> list[dict[str, Any]]:
    """Score, filter, and rerank evidence documents by cross-modal consistency.

    Scoring formula (PDF-aligned):
        S(Ei, q) = α · R(Ei, q) + β · η(q) · C_clip_norm(Ei)

    η(q) = gamma_I when use_adaptive=True, else 1.0.

    Args:
        evidence_list: Each item must contain keys
            ``doc_id``, ``image``, ``text``, ``I_i``, and ``T_i``.
        scorer: A ready-to-use CrossModalConsistencyScorer instance.
        q_emb: Query embedding used to compute R_base via compute_R_base().
        alpha: Weight for R_base in S. Used when use_adaptive=False.
        beta: Weight for η·C_clip_norm in S. Used when use_adaptive=False.
        gamma_I: Image weight for compute_R_base() and (when use_adaptive=True)
                 for compute_eta() and get_adaptive_weights(). Defaults to 0.5.
        gamma_T: Text weight for compute_R_base() and compute_eta(). Defaults to 0.5.
        tau: Minimum raw c_cross threshold; items below are discarded before
             eta scaling so the gate is independent of query type.
        N: Maximum number of results to return.
        use_adaptive: When True, derive alpha/beta from gamma_I/gamma_T and
                      scale C_clip_norm by η(q).

    Returns:
        Up to N items sorted by descending S, each containing
        ``doc_id``, ``image``, ``text``, ``R_base``, ``c_cross``, and ``S``.
        ``c_cross`` is the raw combined score used for tau filtering.
    """
    gamma_I = gamma_I if gamma_I is not None else 0.5
    gamma_T = gamma_T if gamma_T is not None else 0.5

    # Step 1: compute raw scores and R_base for every evidence item
    scored: list[dict[str, Any]] = []
    for ev in evidence_list:
        score_result = scorer.score_document(
            query=None,
            d_i={"id": ev["doc_id"]},
            image_i=ev["image"],
            text_i=ev["text"],
            doc_id=ev["doc_id"],
        )
        R_base = compute_R_base(q_emb, ev["I_i"], ev["T_i"], gamma_I, gamma_T)
        scored.append({
            "doc_id": ev["doc_id"],
            "image": ev["image"],
            "text": ev["text"],
            "R_base": R_base,
            "c_cross": score_result["c_cross"],  # used for tau gate
            "c_clip": score_result["c_clip"],     # used in S formula
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

    # Step 3: resolve alpha / beta
    alpha, beta = get_adaptive_weights(
        alpha=alpha,
        beta=beta,
        gamma_I=gamma_I,
        gamma_T=gamma_T,
        use_adaptive=use_adaptive,
    )

    # Step 4: apply η to C_clip_norm only (PDF: S = α·R + β·η·C_clip_norm)
    eta = compute_eta(gamma_I, gamma_T) if use_adaptive else 1.0
    for item in scored:
        c_clip_scaled = eta * item["c_clip"]
        item["S"] = compute_reranking_score(item["R_base"], c_clip_scaled, alpha, beta)
        del item["c_clip"]  # keep output schema clean; c_cross is already stored

    # Step 5: sort descending by S, return top N
    scored.sort(key=lambda x: x["S"], reverse=True)
    return scored[:N]
