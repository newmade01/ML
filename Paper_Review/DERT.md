# DERT
End to End object Detection , ECCV 2020 발표 (facebook research Team)

------------
##### Detection Transformer(이분매칭,bipartitate matching + Transformer)
(transformer은 보통 자연어 처리에서 사용됨)

- 특징: simple, 좋은 성능

- 기존 논문 문제점: Prior knolege의 요구가 너무 많아 복잡함(bounding box의 형태, 겹칠시 처리법 등) -> NMS 기술(불필요한 박스를 압축, 제거하여 1개의 bounding box만 나올 수 있도록)
1. 입력: CNN 이미지에서 feature를 찾아냄
2. Transformer Encoding -> Transformer Decoding
3. 결과: 클래스 & Bounding Box의 위치를 찾아줌 (없을때, no object )

### 이분매칭: 중복X , 순서가 상관 X(위치 상관 없음)