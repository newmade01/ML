### DERT
End to End object Detection , ECCV 2020 발표 (facebook research Team)

------------
##### Detection Transformer(이분매칭 + Transformer)
(transformer은 보통 자연어 처리에서 사용됨)

- 특징: simple, 좋은 성능
1. 입력: CNN 이미지에서 feature를 찾아냄
2. Transformer Encoding -> Transformer Decoding
3. 결과: 클래스 & Bounding Box의 위치를 찾아줌 (없을때, no object )