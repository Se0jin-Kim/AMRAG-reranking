from __future__ import annotations

from amrag.cross_modal_consistency.types import (
    CrossModalConsistencyConfig,
    CrossModalConsistencyScore,
)


def build_score_output(
    *,
    doc_id: str,
    c_cross: float,
    c_clip: float,
    c_itm: float,
    config: CrossModalConsistencyConfig,
    device: str,
    itm_normalization: str,
) -> CrossModalConsistencyScore:
    return CrossModalConsistencyScore(
        doc_id=doc_id,
        c_cross=float(c_cross),
        c_clip=float(c_clip),
        c_itm=float(c_itm),
        metadata={
            "model_names": {
                "clip": config.clip_model_name,
                "blip_itm": config.blip_model_name,
            },
            "device": device,
            "normalization_method": {
                "clip": config.clip_normalization,
                "itm": itm_normalization,
            },
            "alpha": config.alpha,
            "beta": config.beta,
            "w_text_w_image_used": False,
        },
    )
