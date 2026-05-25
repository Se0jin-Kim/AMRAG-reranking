"""MRAG-Bench 데이터 로더.

각 샘플을 AMRAG 파이프라인이 바로 소비할 수 있는 형태로 변환합니다.

evidence_list 구성
------------------
1. retrieved_images (doc_id="ret_{i}", is_gt=False)
   - CLIP이 검색해온 이미지
   - content 해시가 gt_images와 겹치는 경우 is_gt=True 로 마킹
2. gt_images 중 retrieved에 없는 것만 추가 (doc_id="gt_{i}", is_gt=True)
   - 중복 방지를 위해 pixel 수준 fingerprint 비교

gt_doc_ids
----------
evidence_list 내 is_gt=True 인 항목들의 doc_id 목록.
Recall 계산 시 E_star의 doc_id 또는 이미지 content 와 비교합니다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _fingerprint(img: Image.Image) -> bytes:
    """픽셀 byte fingerprint — 이미지 동일성 비교에 사용."""
    return np.array(img.convert("RGB")).tobytes()


def load_mrag_bench(
    split: str = "test",
    max_samples: int | None = None,
) -> list[dict]:
    """MRAG-Bench를 로드하여 구조화된 샘플 리스트를 반환합니다.

    반환 값 (sample dict 키):
        id            str            샘플 고유 ID (없으면 index)
        query_text    str            질문 텍스트
        query_image   Image.Image    쿼리 이미지
        evidence_list list[dict]     retrieved + gt 이미지 목록
                      각 항목: doc_id, image, text, is_gt
        gt_doc_ids    list[str]      gt 문서의 doc_id 목록
        gt_images     list[Image]    원본 gt 이미지 (content 비교용)
        answer        str            정답 선택지 (A/B/C/D)
        scenario      str            Perspective / Transformative / Others 등
        aspect        str            세부 aspect

    Parameters
    ----------
    split : str
        HuggingFace 데이터셋 split 이름 (기본값: "test")
    max_samples : int | None
        로드할 최대 샘플 수. None이면 전체 로드 (1353개)
    """
    import importlib
    import os
    import sys

    # python tests/eval_mrag_bench.py 실행 시 Python 이 tests/ 를 sys.path[0] 에 추가합니다.
    # tests/datasets/__init__.py 가 존재하므로 'datasets' 임포트가 이 로컬 패키지를
    # HuggingFace datasets 패키지 대신 찾게 됩니다.
    # → tests/ 를 임시로 sys.path 에서 제거하여 실제 패키지를 임포트합니다.
    #   mrag_bench_loader.py 의 존재로 로컬 tests/ 경로를 정확하게 식별합니다.
    _local_parents = [
        p for p in sys.path
        if os.path.isfile(os.path.join(p, "datasets", "mrag_bench_loader.py"))
    ]
    for _p in _local_parents:
        sys.path.remove(_p)
    # 캐시에 로컬 버전이 남아있으면 제거
    if "datasets" in sys.modules and not hasattr(sys.modules["datasets"], "load_dataset"):
        del sys.modules["datasets"]
    try:
        _hf = importlib.import_module("datasets")
        load_dataset = _hf.load_dataset
    finally:
        sys.path.extend(_local_parents)

    dataset = load_dataset("uclanlp/MRAG-Bench", split=split)
    samples: list[dict] = []

    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        query_image: Image.Image = item["image"].convert("RGB")
        query_text: str = item["question"]
        retrieved_images: list[Image.Image] = [
            img.convert("RGB") for img in item["retrieved_images"]
        ]
        gt_images: list[Image.Image] = [
            img.convert("RGB") for img in item["gt_images"]
        ]

        # gt fingerprint 집합 (retrieved와의 overlap 감지용)
        gt_fps: set[bytes] = {_fingerprint(g) for g in gt_images}

        # ── evidence_list 구성 ──────────────────────────
        evidence_list: list[dict] = []
        gt_doc_ids: list[str] = []
        seen_fps: set[bytes] = set()

        # 1) retrieved_images (CLIP 검색 순서 유지)
        for j, img in enumerate(retrieved_images):
            doc_id = f"ret_{j}"
            fp = _fingerprint(img)
            is_gt = fp in gt_fps
            evidence_list.append({
                "doc_id": doc_id,
                "image": img,
                "text": "",       # MRAG-Bench: 텍스트 없음
                "is_gt": is_gt,
            })
            seen_fps.add(fp)
            if is_gt:
                gt_doc_ids.append(doc_id)

        # 2) gt_images 중 retrieved에 없는 것만 추가
        for j, img in enumerate(gt_images):
            fp = _fingerprint(img)
            if fp not in seen_fps:
                doc_id = f"gt_{j}"
                evidence_list.append({
                    "doc_id": doc_id,
                    "image": img,
                    "text": "",
                    "is_gt": True,
                })
                gt_doc_ids.append(doc_id)
                seen_fps.add(fp)

        samples.append({
            "id": item.get("id", str(i)),
            "query_text": query_text,
            "query_image": query_image,
            "evidence_list": evidence_list,
            "gt_doc_ids": gt_doc_ids,
            "gt_images": gt_images,
            "answer": item.get("answer_choice", item.get("answer", "")),
            "scenario": item.get("scenario", "Unknown"),
            "aspect": item.get("aspect", "Unknown"),
        })

    return samples
