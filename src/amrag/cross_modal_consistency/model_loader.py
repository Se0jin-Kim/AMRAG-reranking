from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from amrag.cross_modal_consistency.types import CrossModalConsistencyConfig


@dataclass(frozen=True)
class ModelBundle:
    clip_model: Any
    clip_processor: Any
    blip_model: Any
    blip_processor: Any
    device: str


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for cross-modal consistency scoring. "
            "Install the project dependencies before constructing the scorer."
        ) from exc
    return torch


def resolve_device(device: str) -> str:
    torch = import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def load_model_bundle(config: CrossModalConsistencyConfig) -> ModelBundle:
    import_torch()
    try:
        from transformers import AutoProcessor, BlipForImageTextRetrieval, CLIPModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load CLIP and BLIP-ITM models. "
            "Install the project dependencies before constructing the scorer."
        ) from exc

    device = resolve_device(config.device)
    clip_processor = AutoProcessor.from_pretrained(config.clip_model_name)
    clip_model = CLIPModel.from_pretrained(config.clip_model_name).to(device).eval()

    blip_processor = AutoProcessor.from_pretrained(config.blip_model_name)
    blip_model = BlipForImageTextRetrieval.from_pretrained(
        config.blip_model_name
    ).to(device).eval()

    return ModelBundle(
        clip_model=clip_model,
        clip_processor=clip_processor,
        blip_model=blip_model,
        blip_processor=blip_processor,
        device=device,
    )
