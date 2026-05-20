from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_prompt(
    query_text: str,
    query_image: Any,
    E_star: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Build a LLaVA-OneVision conversation prompt from reranked evidence.

    Args:
        query_text:  The original user question.
        query_image: Query image (PIL.Image).
        E_star:      Reranked evidence list from rerank(). Each item must
                     contain ``image`` and ``text``.

    Returns:
        messages: Single-turn conversation list in LLaVA-OneVision format.
        images:   Flat image list — query image first, then evidence images
                  in the same order as E_star.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be a non-empty string.")

    if query_image is None:
        raise ValueError("query_image must not be None.")

    if not E_star:
        logger.warning("build_prompt: E_star is empty — generating prompt with no evidence context.")

    image_placeholders: list[dict[str, str]] = [{"type": "image"}] * (1 + len(E_star))

    context_lines = "\n".join(
        f"[{i + 1}] {ev['text']}" for i, ev in enumerate(E_star)
    )
    text_block = (
        f"Question: {query_text}\n\n"
        f"Reference Context:\n{context_lines}\n\n"
        "Based on the above context and images, please answer the question."
    )

    messages = [
        {
            "role": "user",
            "content": [
                *image_placeholders,
                {"type": "text", "text": text_block},
            ],
        }
    ]

    images = [query_image] + [ev["image"] for ev in E_star]

    return messages, images
