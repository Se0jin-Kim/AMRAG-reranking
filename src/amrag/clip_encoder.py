"""CLIP embedding encoder for the AMRAG pipeline."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """CLIP 기반 이미지/텍스트 임베딩 계산기.

    encode_image, encode_text 모두 L2 정규화된 벡터를 반환하므로
    rerank() 내부의 dot product가 cosine similarity와 동치가 됩니다.

    encode_text("") 는 zero vector를 반환하여 텍스트가 없는 문서
    (MRAG-Bench 등)의 T_i 로 바로 사용할 수 있습니다.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        # projection_dim: 512 for ViT-B/32, 768 for ViT-L/14
        self.dim: int = self.model.config.projection_dim
        print(f"CLIPEncoder: model={model_id}  device={device}  dim={self.dim}")

    # ------------------------------------------------------------------ #
    # 단일 임베딩
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_tensor(output: object) -> torch.Tensor:
        """transformers 버전에 따라 tensor 또는 ModelOutput 을 반환하는 것을
        항상 torch.Tensor 로 통일합니다 (transformers 5.x 대응)."""
        if isinstance(output, torch.Tensor):
            return output
        # BaseModelOutputWithPooling 등 ModelOutput 계열
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(output, attr):
                return getattr(output, attr)
        # tuple fallback
        return output[0]  # type: ignore[index]

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """이미지 → L2 정규화된 CLIP 임베딩 (shape: dim,)."""
        inputs = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            feats = self._to_tensor(self.model.get_image_features(**inputs))
            feats = F.normalize(feats, p=2, dim=-1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """텍스트 → L2 정규화된 CLIP 임베딩 (shape: dim,).

        text가 빈 문자열이거나 공백만 있으면 zero vector를 반환합니다.
        이 경우 dot(q_emb, T_i) = 0 이 되어 S_text 기여분이 0이 됩니다.
        """
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        with torch.inference_mode():
            feats = self._to_tensor(self.model.get_text_features(**inputs))
            feats = F.normalize(feats, p=2, dim=-1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # 융합 쿼리 임베딩
    # ------------------------------------------------------------------ #

    def encode_query(
        self,
        text: str,
        image: Image.Image,
        gamma_I: float = 0.5,
        gamma_T: float = 0.5,
    ) -> np.ndarray:
        """이미지·텍스트를 가중합한 뒤 L2 정규화하여 반환합니다.

            q_emb = L2_norm( gamma_I * encode_image(image)
                           + gamma_T * encode_text(text) )

        text가 빈 문자열이면 encode_text는 zero vector를 반환하므로
        q_emb = encode_image(image) (gamma_I 값에 무관하게 이미지 임베딩만 반영).
        """
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)
        combined = gamma_I * img_emb + gamma_T * txt_emb
        norm = float(np.linalg.norm(combined))
        if norm > 1e-8:
            combined = combined / norm
        return combined.astype(np.float32)
