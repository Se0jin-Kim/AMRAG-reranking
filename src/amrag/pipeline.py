from __future__ import annotations

import logging
from typing import Any

from amrag.cross_modal_consistency import CrossModalConsistencyScorer
from amrag.prompt_builder import build_prompt
from amrag.reranking import rerank

logger = logging.getLogger(__name__)


class AMRAGPipeline:
    """End-to-end AMRAG pipeline: rerank → prompt → generate."""

    def __init__(
        self,
        scorer: CrossModalConsistencyScorer,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        load_model: bool = False,
        device: str = "cuda",
    ) -> None:
        self.scorer = scorer
        self.model_id = model_id
        self.device = device

        if load_model:
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_id
            ).to(device).eval()
        else:
            self.model = None
            self.processor = None

    def run(
        self,
        query_text: str,
        query_image: Any,
        evidence_list: list[dict[str, Any]],
        q_emb: Any,
        gamma_I: float = 0.5,
        gamma_T: float = 0.5,
        alpha: float = 0.7,
        beta: float = 0.3,
        tau: float = 0.5,
        N: int = 5,
        use_adaptive: bool = False,
        lambda_c: float = 0.5,
    ) -> dict[str, Any]:
        if q_emb is None:
            raise ValueError("q_emb must not be None.")

        # Step 1: rerank evidence by cross-modal consistency
        logger.info(
            "pipeline.run: reranking %d evidence items (tau=%.2f, N=%d, use_adaptive=%s)",
            len(evidence_list), tau, N, use_adaptive,
        )
        E_star = rerank(
            evidence_list=evidence_list,
            scorer=self.scorer,
            q_emb=q_emb,
            alpha=alpha,
            beta=beta,
            gamma_I=gamma_I,
            gamma_T=gamma_T,
            tau=tau,
            N=N,
            use_adaptive=use_adaptive,
            lambda_c=lambda_c,
            query_text=query_text,
        )
        logger.info("pipeline.run: E_star size after reranking = %d", len(E_star))

        if not E_star:
            logger.warning("pipeline.run: no evidence survived filtering — skipping generation.")
            return {"answer": None, "E_star": [], "messages": [], "images": []}

        # Step 2: build LLaVA-OneVision prompt
        messages, images = build_prompt(
            query_text=query_text,
            query_image=query_image,
            E_star=E_star,
        )
        logger.info(
            "pipeline.run: prompt built with %d image placeholders.",
            sum(1 for c in messages[0]["content"] if c["type"] == "image"),
        )

        # Step 3: generate answer if model is loaded
        answer: str | None = None
        if self.model is not None and self.processor is not None:
            import torch

            inputs = self.processor(
                text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                images=images,
                return_tensors="pt",
            ).to(self.device)

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs)
                generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
                answer = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                logger.info("pipeline.run: generation succeeded.")
            except Exception:
                logger.exception("pipeline.run: model.generate() failed — returning answer=None.")
                answer = None

        return {
            "answer": answer,
            "E_star": E_star,
            "messages": messages,
            "images": images,
        }
