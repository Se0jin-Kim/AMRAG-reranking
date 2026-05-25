from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping
from typing import Any

from amrag.cross_modal_consistency.model_loader import (
    ModelBundle,
    import_torch,
    load_model_bundle,
    resolve_device,
)
from amrag.cross_modal_consistency.output import build_score_output
from amrag.cross_modal_consistency.preprocess import (
    prepare_blip_inputs,
    prepare_clip_image_inputs,
    prepare_clip_text_inputs,
)
from amrag.cross_modal_consistency.types import (
    CrossModalConsistencyConfig,
    CrossModalConsistencyScore,
    CrossModalInput,
    MissingModalityError,
)


@functools.lru_cache(maxsize=1)
def _load_spacy_nlp() -> Any:
    """Load en_core_web_sm once; return None if spacy or the model is unavailable."""
    try:
        import spacy  # type: ignore[import]
        return spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return None


def _extract_entities(text: str | None) -> set[str]:
    """Return lowercase named-entity strings from text.

    Uses spacy en_core_web_sm when available; falls back to capitalized words.
    Returns an empty set when text is None or empty.
    """
    if not text or not text.strip():
        return set()
    nlp = _load_spacy_nlp()
    if nlp is not None:
        doc = nlp(text)
        return {ent.text.lower() for ent in doc.ents}
    # Fallback: treat capitalized (non-first-position) words as proxy entities
    tokens = text.split()
    return {
        t.strip(".,!?;:'\"").lower()
        for t in tokens
        if t and t[0].isupper() and len(t) > 1
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class CrossModalConsistencyScorer:
    """Scores image-text consistency for retrieved AMRAG documents."""

    def __init__(
        self,
        config: CrossModalConsistencyConfig | None = None,
        *,
        model_bundle: ModelBundle | None = None,
        clip_scorer: Any | None = None,
        itm_scorer: Any | None = None,
    ) -> None:
        self.config = config or CrossModalConsistencyConfig()
        self._validate_config()

        if model_bundle is None and (clip_scorer is None or itm_scorer is None):
            model_bundle = load_model_bundle(self.config)

        self.model_bundle = model_bundle
        self.clip_scorer = clip_scorer
        self.itm_scorer = itm_scorer
        self.device = model_bundle.device if model_bundle is not None else resolve_device(self.config.device)

    def score_document(
        self,
        *,
        query: str | None,
        d_i: Any,
        image_i: Any,
        text_i: str | None,
        w_text: float | None = None,
        w_image: float | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        scoring_input = CrossModalInput(
            query=query,
            d_i=d_i,
            image_i=image_i,
            text_i=self._validate_text(text_i),
            w_text=w_text,
            w_image=w_image,
            doc_id=doc_id,
        )
        return self.score(scoring_input).to_dict()

    def score(self, scoring_input: CrossModalInput) -> CrossModalConsistencyScore:
        self._validate_image(scoring_input.image_i)
        text = self._validate_text(scoring_input.text_i)

        c_clip = self.compute_clip_consistency(scoring_input.image_i, text)
        c_itm, itm_normalization = self._compute_itm_consistency_with_metadata(
            scoring_input.image_i, text
        )
        self._validate_component_score(c_clip, "c_clip")
        self._validate_component_score(c_itm, "c_itm")

        if scoring_input.query is not None:
            q_ents = _extract_entities(scoring_input.query)
            t_ents = _extract_entities(text)
            c_entity = _jaccard(q_ents, t_ents)
        else:
            c_entity = 0.0

        c_cross = (
            self.config.alpha * c_clip
            + self.config.beta * c_itm
            + self.config.gamma * c_entity
        )
        self._validate_component_score(c_cross, "c_cross")
        return build_score_output(
            doc_id=self._resolve_doc_id(scoring_input.d_i, scoring_input.doc_id),
            c_cross=c_cross,
            c_clip=c_clip,
            c_itm=c_itm,
            c_entity=c_entity,
            config=self.config,
            device=self.device,
            itm_normalization=itm_normalization,
        )

    def score_batch(self, inputs: Iterable[CrossModalInput]) -> list[dict[str, Any]]:
        return [self.score(item).to_dict() for item in inputs]

    def compute_clip_consistency(self, image: Any, text: str) -> float:
        if self.clip_scorer is not None:
            return float(self.clip_scorer(image, text))
        if self.model_bundle is None:
            raise RuntimeError("CLIP model bundle is not initialized.")

        torch = import_torch()
        image_inputs = prepare_clip_image_inputs(
            self.model_bundle.clip_processor, [image], self.device
        )
        text_inputs = prepare_clip_text_inputs(
            self.model_bundle.clip_processor, [text], self.device
        )
        with torch.inference_mode():
            image_features = self.model_bundle.clip_model.get_image_features(**image_inputs)
            text_features = self.model_bundle.clip_model.get_text_features(**text_inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            cosine = (image_features * text_features).sum(dim=-1).squeeze(0)
            if self.config.clip_normalization == "cosine_l2_rescaled_0_1":
                score = (cosine + 1.0) / 2.0
            else:
                raise ValueError(
                    f"Unsupported CLIP normalization: {self.config.clip_normalization}"
                )
        return float(score.detach().cpu().item())

    def compute_itm_consistency(self, image: Any, text: str) -> float:
        score, _normalization = self._compute_itm_consistency_with_metadata(image, text)
        return score

    def _compute_itm_consistency_with_metadata(self, image: Any, text: str) -> tuple[float, str]:
        if self.itm_scorer is not None:
            return float(self.itm_scorer(image, text)), "injected_scorer"
        if self.model_bundle is None:
            raise RuntimeError("BLIP-ITM model bundle is not initialized.")

        torch = import_torch()
        inputs = prepare_blip_inputs(self.model_bundle.blip_processor, [image], [text], self.device)
        with torch.inference_mode():
            outputs = self.model_bundle.blip_model(**inputs, use_itm_head=True)
            if not hasattr(outputs, "itm_score"):
                raise RuntimeError("BLIP-ITM output did not include itm_score.")
            itm_score = outputs.itm_score
            if itm_score.ndim == 1:
                if itm_score.numel() == 1:
                    matched_probability = torch.sigmoid(itm_score)[0]
                    normalization = "sigmoid_matched_probability"
                elif itm_score.numel() > self.config.itm_matched_index:
                    probabilities = torch.softmax(itm_score, dim=-1)
                    matched_probability = probabilities[self.config.itm_matched_index]
                    normalization = "softmax_matched_probability"
                else:
                    raise RuntimeError("BLIP-ITM itm_score does not include matched class.")
            elif itm_score.shape[-1] == 1:
                matched_probability = torch.sigmoid(itm_score.squeeze(-1))[0]
                normalization = "sigmoid_matched_probability"
            else:
                probabilities = torch.softmax(itm_score, dim=-1)
                matched_probability = probabilities[0, self.config.itm_matched_index]
                normalization = "softmax_matched_probability"
        return float(matched_probability.detach().cpu().item()), normalization

    def _validate_config(self) -> None:
        alpha = self.config.alpha
        beta = self.config.beta
        gamma = self.config.gamma
        if not math.isfinite(alpha) or not math.isfinite(beta) or not math.isfinite(gamma):
            raise ValueError("alpha, beta, and gamma must be finite numbers.")
        if alpha < 0.0 or beta < 0.0 or gamma < 0.0:
            raise ValueError("alpha, beta, and gamma must be non-negative.")
        if self.config.itm_matched_index < 0:
            raise ValueError("itm_matched_index must be non-negative.")
        if self.config.validate_alpha_beta and not math.isclose(
            alpha + beta + gamma,
            1.0,
            rel_tol=0.0,
            abs_tol=self.config.alpha_beta_tolerance,
        ):
            raise ValueError(
                f"alpha + beta + gamma must equal 1.0, "
                f"got alpha={alpha}, beta={beta}, gamma={gamma}."
            )

    @staticmethod
    def _validate_image(image: Any) -> None:
        if image is None:
            raise MissingModalityError("image_i is required for cross-modal consistency scoring.")

    @staticmethod
    def _validate_text(text: str | None) -> str:
        if text is None or not str(text).strip():
            raise MissingModalityError("text_i is required for cross-modal consistency scoring.")
        return str(text)

    @staticmethod
    def _validate_component_score(score: float, name: str) -> None:
        if not math.isfinite(score):
            raise ValueError(f"{name} must be a finite number.")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"{name} must be in the [0.0, 1.0] range.")

    @staticmethod
    def _resolve_doc_id(document: Any, explicit_doc_id: str | None) -> str:
        if explicit_doc_id is not None:
            return str(explicit_doc_id)
        if isinstance(document, Mapping):
            for key in ("doc_id", "id"):
                if key in document and document[key] is not None:
                    return str(document[key])
        for key in ("doc_id", "id"):
            value = getattr(document, key, None)
            if value is not None:
                return str(value)
        return ""
