from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CrossModalConsistencyError(ValueError):
    """Base exception for invalid cross-modal consistency scoring inputs."""


class MissingModalityError(CrossModalConsistencyError):
    """Raised when an image or text modality required for scoring is missing."""


@dataclass(frozen=True)
class CrossModalConsistencyConfig:
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    validate_alpha_beta: bool = True
    alpha_beta_tolerance: float = 1e-6
    device: str = "auto"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    blip_model_name: str = "Salesforce/blip-itm-base-coco"
    clip_normalization: str = "cosine_l2_rescaled_0_1"
    itm_matched_index: int = 1


@dataclass(frozen=True)
class CrossModalInput:
    query: str | None
    d_i: Any
    image_i: Any
    text_i: str
    w_text: float | None = None
    w_image: float | None = None
    doc_id: str | None = None


@dataclass(frozen=True)
class CrossModalConsistencyScore:
    doc_id: str
    c_cross: float
    c_clip: float
    c_itm: float
    c_entity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "c_cross": self.c_cross,
            "c_clip": self.c_clip,
            "c_itm": self.c_itm,
            "c_entity": self.c_entity,
            "metadata": self.metadata,
        }
