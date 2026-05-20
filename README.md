# AMRAG: Adaptive Multimodal Retrieval-Augmented Generation image-text cross-modal consistency scoring

## Overview

AMRAG: image-text cross-modal consistency scoring

검색된 문서 후보의 이미지와 텍스트 사이 cross-modal consistency score를 계산하는 모듈임.

현재 구현 범위: retrieved document에 대한 consistency scoring
최종 reranking, filtering threshold 적용, query-adaptive weighting, generation logic은 이 모듈에서 수행하지 않음.


`최종 return: C_cross(d_i) -> dict`

## Scoring Formula

문서 후보 `d_i`에 대한 최종 cross-modal consistency score는 다음과 같이 계산함.

```text
C_cross(d_i) = alpha * C_clip(d_i) + beta * C_itm(d_i)
```

각 element의 의미

- `C_clip(d_i)`: CLIP 기반 image-text similarity score
- `C_itm(d_i)`: BLIP-ITM 기반 image-text matching score
- `alpha`, `beta`: 두 component score를 결합하기 위한 가중치
    - types.py -> CrossModalConsistencyConfig 에서 값 변경

`w_text`, `w_image`는 인터페이스 호환을 위해 받을 수 있으나, 현재 `C_cross(d_i)` 계산에는 사용하지 않음.

## Installation

프로젝트 루트에서 editable install을 수행함.

```powershell
python -m pip install -e .
```

테스트 의존성까지 설치하려면 다음 명령을 사용함.

```powershell
python -m pip install -e ".[test]"
```

## Usage

문서 하나에 대한 score 계산 예시는 다음과 같음.

```python
from amrag.cross_modal_consistency import CrossModalConsistencyScorer

scorer = CrossModalConsistencyScorer()

result = scorer.score_document(
    query="What animal is shown?",
    d_i={"id": "doc-1"},
    image_i=image,
    text_i="A cat sitting on a mat.",
    w_text=0.7,
    w_image=0.3,
)
```

검색된 문서 후보 `d_1, d_2, ..., d_k`에 대해 각 후보별 score를 계산하는 예시는 다음과 같음.

```python
from amrag.cross_modal_consistency import CrossModalConsistencyScorer

scorer = CrossModalConsistencyScorer()
results = []

for d_i in retrieved_docs:
    result = scorer.score_document(
        query=query,
        d_i=d_i,
        image_i=d_i["image"],
        text_i=d_i["text"],
        w_text=w_text,
        w_image=w_image,
    )
    results.append(result)
```

## Inputs

`score_document()`의 주요 입력은 다음과 같음.

- `query`: 원본 사용자 query
- `d_i`: 검색된 문서 후보 객체
- `image_i`: 문서 후보에 연결된 이미지
- `text_i`: 문서 후보에 연결된 텍스트
- `w_text`: 외부 retrieval 단계에서 계산된 text weight
- `w_image`: 외부 retrieval 단계에서 계산된 image weight
- `doc_id`: 명시적으로 전달할 문서 ID

`w_text + w_image = 1`을 가정할 수 있으나, 현재 scoring 계산에는 사용하지 않음.

## Outputs

`score_document()`의 반환값은 `dict` 형태임.

```python
{
    "doc_id": "doc-1",
    "c_cross": 0.7,
    "c_clip": 0.8,
    "c_itm": 0.6,
    "metadata": {
        "model_names": {
            "clip": "openai/clip-vit-base-patch32",
            "blip_itm": "Salesforce/blip-itm-base-coco",
        },
        "device": "cpu",
        "normalization_method": {
            "clip": "cosine_l2_rescaled_0_1",
            "itm": "softmax_matched_probability",
        },
        "alpha": 0.5,
        "beta": 0.5,
        "w_text_w_image_used": False,
    },
}
```

주요 score field는 다음과 같음.

- `c_cross`: 최종 cross-modal consistency score임
- `c_clip`: CLIP 기반 component score임
- `c_itm`: BLIP-ITM 기반 component score임
- `metadata`: 모델명, device, normalization method, config 정보를 담는 field임

## Extracting Scores

단일 결과에서 score를 추출하는 예시는 다음과 같음.

```python
c_cross = result["c_cross"]
c_clip = result["c_clip"]
c_itm = result["c_itm"]
```

여러 문서 후보 결과에서 `c_cross`만 추출하는 예시는 다음과 같음.

```python
c_cross_scores = [result["c_cross"] for result in results]
```

이 모듈은 score를 반환할 뿐이며, score를 이용한 reranking, filtering, top-n selection은 외부 모듈의 책임임.

## External Filtering And Top-N Example

다음 코드는 외부 시스템에서 `c_cross`를 사용해 filtering 및 top-n 정렬을 수행하는 예시임.
해당 로직은 이 모듈 내부 구현 범위에 포함되지 않음.

```python
from amrag.cross_modal_consistency import CrossModalConsistencyScorer

scorer = CrossModalConsistencyScorer()

scored_docs = []
for d_i in retrieved_docs:
    score_result = scorer.score_document(
        query=query,
        d_i=d_i,
        image_i=d_i["image"],
        text_i=d_i["text"],
        w_text=w_text,
        w_image=w_image,
    )
    scored_docs.append(
        {
            "document": d_i,
            "score": score_result,
        }
    )

threshold = 0.6
top_n = 5

filtered_docs = [
    item
    for item in scored_docs
    if item["score"]["c_cross"] >= threshold
]

top_n_docs = sorted(
    filtered_docs,
    key=lambda item: item["score"]["c_cross"],
    reverse=True,
)[:top_n]
```

`top_n_docs`의 각 item은 원본 문서와 score 결과를 함께 포함함.

```python
for item in top_n_docs:
    document = item["document"]
    c_cross = item["score"]["c_cross"]
```

## Configuration

`CrossModalConsistencyConfig`로 scoring 설정을 변경할 수 있음.

```python
from amrag.cross_modal_consistency import (
    CrossModalConsistencyConfig,
    CrossModalConsistencyScorer,
)

config = CrossModalConsistencyConfig(
    alpha=0.5,
    beta=0.5,
    device="cpu",
)

scorer = CrossModalConsistencyScorer(config)
```

주요 설정값은 다음과 같음.

- `alpha`: `C_clip`에 적용되는 가중치임
- `beta`: `C_itm`에 적용되는 가중치임
- `device`: `"auto"`, `"cpu"`, `"cuda"` 등 model 실행 device 설정임
- `clip_model_name`: CLIP model name임
- `blip_model_name`: BLIP-ITM model name임
- `clip_normalization`: CLIP score normalization method임
- `itm_matched_index`: ITM matched class index임

## Project Structure

프로젝트 구조는 다음과 같음.

```text
src/amrag/cross_modal_consistency/
  __init__.py       # public import interface 정의함
  scoring.py        # main scorer logic 구현함
  types.py          # dataclass, config, exception 정의함
  output.py         # output formatting 구현함
  preprocess.py     # model input preprocessing 구현함
  model_loader.py   # model loading 및 device resolution 구현함

tests/
  conftest.py
  test_cross_modal_consistency.py
```

## Tests

프로젝트 루트에서 pytest로 테스트를 실행함.

```powershell
python -m pytest
```

특정 테스트 파일만 실행하려면 다음 명령을 사용함.

```powershell
python -m pytest tests/test_cross_modal_consistency.py
```
