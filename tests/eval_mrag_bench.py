"""
MRAG-Bench reranking 효과 평가 스크립트.

평가 흐름
---------
1. MRAG-Bench 로드        (mrag_bench_loader)
2. CLIPEncoder 로 임베딩   (amrag.clip_encoder)
3. AMRAGPipeline.run() 으로 reranking
4. Recall@K 계산           (K=1, 3, 5)
5. 결과 출력

두 가지 설정 비교
-----------------
- Baseline : 별도 reranking 없이 CLIP 검색 순서(retrieved_images) 그대로
- 팀 B     : AMRAGPipeline.run() 으로 reranking 후 상위 N개

Baseline 풀 : retrieved 이미지 (5개, CLIP 순서)
팀 B 풀     : retrieved + gt 이미지 전체 (combined evidence_list)
Recall 계산 : gt_images와 content (pixel bytes) 비교

실행 방법
---------
    cd AMRAG-Cross-modal-consistency
    python tests/eval_mrag_bench.py

주요 하이퍼파라미터
-------------------
    GAMMA_I     = 0.7   (이미지 중심 벤치마크)
    GAMMA_T     = 0.3
    TAU         = 0.0   (텍스트 없으므로 c_cross 필터링 비활성화)
    N           = 5
    LAMBDA_C    = 0.0   (c_cross 항 비활성화, 텍스트 없음)
    MAX_SAMPLES = 50    (테스트용 소규모; None 이면 전체 1353개)

주의사항
--------
- MRAG-Bench 문서에는 텍스트가 없음 → text_i = ""
- CLIPEncoder.encode_text("") 는 zero vector 반환
- T_i = zeros → dot(q_emb, T_i) = 0 → S_text 기여 없음
- CrossModalConsistencyScorer 는 빈 text_i 에서 에러 발생 →
  c_cross = 0.0 을 항상 반환하는 NullScorer 로 대체
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ── 경로 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# NOTE: tests/datasets/ 를 sys.path 에 직접 추가하면
#       'datasets' 라는 이름이 HuggingFace datasets 패키지를 shadow 합니다.
# → importlib 으로 mrag_bench_loader 를 직접 로드하여 충돌을 방지합니다.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib.util as _ilu  # noqa: E402

_loader_path = ROOT / "tests" / "datasets" / "mrag_bench_loader.py"
_spec = _ilu.spec_from_file_location("mrag_bench_loader", _loader_path)
_mod = _ilu.module_from_spec(_spec)            # type: ignore[arg-type]
_spec.loader.exec_module(_mod)                 # type: ignore[union-attr]
load_mrag_bench = _mod.load_mrag_bench

from amrag.clip_encoder import CLIPEncoder     # noqa: E402
from amrag.pipeline import AMRAGPipeline       # noqa: E402

# ── 하이퍼파라미터 ─────────────────────────────────────────
GAMMA_I = 0.7
GAMMA_T = 0.3
TAU = 0.0
N = 5
LAMBDA_C = 0.0
MAX_SAMPLES = None


# ──────────────────────────────────────────────────────────
# NullScorer
# CrossModalConsistencyScorer 는 빈 text_i 에서 MissingModalityError 를
# 발생시킵니다. MRAG-Bench 는 텍스트가 없으므로 이 placeholder 를 사용합니다.
# ──────────────────────────────────────────────────────────
class NullScorer:
    """c_cross = 0.0 을 반환하는 scorer. 텍스트 없는 데이터셋 전용."""

    def score_document(
        self,
        *,
        query: str | None,       # noqa: ARG002
        d_i: Any,
        image_i: Any,            # noqa: ARG002
        text_i: str | None,      # noqa: ARG002
        doc_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resolved_id = doc_id if doc_id is not None else str(d_i.get("id", ""))
        return {
            "doc_id": resolved_id,
            "c_cross": 0.0,
            "c_clip": 0.0,
            "c_itm": 0.0,
            "c_entity": 0.0,
            "metadata": {},
        }


# ──────────────────────────────────────────────────────────
# Recall 계산
# ──────────────────────────────────────────────────────────
def _img_bytes(img: Image.Image) -> bytes:
    return np.array(img.convert("RGB")).tobytes()


def recall_at_k(
    result_images: list[Image.Image],
    gt_images: list[Image.Image],
    k: int,
) -> float:
    """상위 K개 중 gt_image 가 하나라도 포함되면 1.0, 아니면 0.0."""
    if not gt_images or not result_images:
        return 0.0
    gt_keys = {_img_bytes(g) for g in gt_images}
    return 1.0 if any(_img_bytes(r) in gt_keys for r in result_images[:k]) else 0.0


# ──────────────────────────────────────────────────────────
# 메인 평가 루프
# ──────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 58)
    print("  MRAG-Bench Reranking 효과 평가")
    print("=" * 58)

    # 1. 데이터 로드 ─────────────────────────────────────
    print(f"\n[1/3] MRAG-Bench 로드 (split=test, max_samples={MAX_SAMPLES}) ...")
    samples = load_mrag_bench(split="test", max_samples=MAX_SAMPLES)
    total = len(samples)
    print(f"      {total}개 샘플 로드 완료.")

    # 2. 모델 초기화 ─────────────────────────────────────
    print("\n[2/3] CLIPEncoder / AMRAGPipeline 초기화 ...")
    encoder = CLIPEncoder()                  # 자동 device 선택
    null_scorer = NullScorer()

    # AMRAGPipeline: load_model=False → 생성 없이 reranking만 수행
    pipeline = AMRAGPipeline(scorer=null_scorer, load_model=False)

    # 3. 평가 ────────────────────────────────────────────
    print(f"\n[3/3] {total}개 샘플 평가 시작 ...")

    baseline_totals: dict[int, float] = {1: 0.0, 3: 0.0, 5: 0.0}
    teamb_totals:    dict[int, float] = {1: 0.0, 3: 0.0, 5: 0.0}
    scenario_b1:  dict[str, list[float]] = defaultdict(list)
    scenario_t1:  dict[str, list[float]] = defaultdict(list)

    for idx, sample in enumerate(samples):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1:>3d}/{total}]")

        query_text   = sample["query_text"]
        query_image  = sample["query_image"]
        evidence_list = sample["evidence_list"]
        gt_images    = sample["gt_images"]
        scenario     = sample["scenario"]

        # ── 임베딩 ──────────────────────────────────────
        # 쿼리: gamma_I * image_emb + gamma_T * text_emb (L2 정규화)
        q_emb = encoder.encode_query(
            text=query_text,
            image=query_image,
            gamma_I=GAMMA_I,
            gamma_T=GAMMA_T,
        )

        # evidence 임베딩 주입
        # I_i: L2 정규화된 CLIP 이미지 임베딩
        # T_i: encode_text("") = zeros → S_text = dot(q_emb, 0) = 0
        for ev in evidence_list:
            ev["I_i"] = encoder.encode_image(ev["image"])
            ev["T_i"] = encoder.encode_text(ev["text"])  # "" → zeros

        # ── Baseline: CLIP 원래 순서 (retrieved 이미지만) ─
        # doc_id 가 "ret_" 로 시작하는 항목 = retrieved_images 순서 그대로
        baseline_images = [
            ev["image"] for ev in evidence_list if ev["doc_id"].startswith("ret_")
        ]

        for k in (1, 3, 5):
            baseline_totals[k] += recall_at_k(baseline_images, gt_images, k)
        scenario_b1[scenario].append(recall_at_k(baseline_images, gt_images, 1))

        # ── 팀 B: AMRAGPipeline reranking ──────────────
        # S = gamma_I * dot(q_emb, I_i)
        #   + gamma_T * dot(q_emb, T_i=0)    = 0
        #   + lambda_c=0 * c_cross=0          = 0
        # → S = 0.7 * sim(query_image, evidence_image)
        result = pipeline.run(
            query_text=query_text,
            query_image=query_image,
            evidence_list=evidence_list,
            q_emb=q_emb,
            gamma_I=GAMMA_I,
            gamma_T=GAMMA_T,
            tau=TAU,
            N=N,
            lambda_c=LAMBDA_C,
        )
        teamb_images = [item["image"] for item in result["E_star"]]

        for k in (1, 3, 5):
            teamb_totals[k] += recall_at_k(teamb_images, gt_images, k)
        scenario_t1[scenario].append(recall_at_k(teamb_images, gt_images, 1))

    # ── 결과 출력 ─────────────────────────────────────────
    print(f"\n{'=' * 58}")
    print(f"=== MRAG-Bench 평가 결과 ({total} samples) ===")
    print(f"{'=' * 58}\n")

    print("[Baseline - No Reranking]")
    for k in (1, 3, 5):
        pct = baseline_totals[k] / total * 100
        print(f"Recall@{k}: {pct:.1f}%")

    print(f"\n[팀 B - Consistency Reranking]")
    for k in (1, 3, 5):
        pct = teamb_totals[k] / total * 100
        print(f"Recall@{k}: {pct:.1f}%")

    print(f"\n--- Scenario별 Recall@1 ---")
    all_scenarios = sorted(set(scenario_b1) | set(scenario_t1))
    for sc in all_scenarios:
        b_hits = scenario_b1[sc]
        t_hits = scenario_t1[sc]
        b_pct = sum(b_hits) / len(b_hits) * 100 if b_hits else 0.0
        t_pct = sum(t_hits) / len(t_hits) * 100 if t_hits else 0.0
        n = len(b_hits)
        print(f"{sc:<22s}: Baseline {b_pct:5.1f}% → 팀 B {t_pct:5.1f}%  (n={n})")

    print()


if __name__ == "__main__":
    main()
