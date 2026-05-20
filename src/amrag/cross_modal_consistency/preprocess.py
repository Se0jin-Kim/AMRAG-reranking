from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def move_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def prepare_clip_image_inputs(processor: Any, images: list[Any], device: str) -> dict[str, Any]:
    inputs = processor(images=images, return_tensors="pt")
    return move_to_device(inputs, device)


def prepare_clip_text_inputs(processor: Any, texts: list[str], device: str) -> dict[str, Any]:
    inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
    return move_to_device(inputs, device)


def prepare_blip_inputs(
    processor: Any,
    images: list[Any],
    texts: list[str],
    device: str,
) -> dict[str, Any]:
    inputs = processor(
        images=images,
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return move_to_device(inputs, device)
