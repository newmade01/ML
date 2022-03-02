# DERT
End to End object Detection , ECCV 2020 발표 (facebook research Team)

------------
##### Detection Transformer(이분매칭,bipartitate matching + Transformer)
(transformer은 보통 자연어 처리에서 사용됨)

- 특징: simple, 좋은 성능

- 기존 논문 문제점: Prior knolege의 요구가 너무 많아 복잡함(bounding box의 형태, 겹칠시 처리법 등) -> NMS 기술(불필요한 박스를 압축, 제거하여 1개의 bounding box만 나올 수 있도록)

### 순서
1. 입력: CNN 이미지에서 feature를 찾아냄
2. Transformer Encoding -> Transformer Decoding
3. 결과: 클래스 & Bounding Box의 위치를 찾아줌 (없을때, no object )

### 이분매칭
- set Prediction Problem 해결 
- 중복X 
- 순서가 상관 X(위치 상관 없음)
- 직접적으로 문제를 해결
- 몇개의 instance의 box가 나오는지 output에 미리 고정을 시켜놓음
- 하나씩 일대일 매칭(instance가 중복되지 않도록 )
- 전체 loss값이 줄어들도록 매칭 

### Transformer

- Attention: 문맥 정보를 이해
- 픽셀들이 Attention 상호작용을 이
- 거리가 먼 픽셀간의 파악이 쉬
- Encoder: feature이 담긴 각 픽셀의 위치 데이터 입력
- Decoder: N개의 (=output값)object query 초기입력, 고유한 class&bounding box로 구분됨
- 