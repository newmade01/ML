# YOLO
- object detection 객체탐지
### 객체 탐지
- bounding box를 탐지, 이미지의 객체값 표시(위치), 객체의 class를 알려줌
- 신뢰도 출력
- 사용도: 자율주행(차, 보행자, 표지판), 의료(x-ray, 방사선 분석), 제조업(로봇 조립, 수리), 보안 산업(위협, 사람, 범죄 발견)
- Bounding Box
  1. IOU(intersection Over Union): 모델의 성능을 나타냄
     - 실측과 예측값이 얼마나 겹치는지 지표
     - 오버랩 부분/전체합 => 교집합 해당부분
     - Poor(45%) < Good(73%) < Ecellent(92%)
  2. NMS(Non-Maximum Suppression, 비최댓값 억제)
     - 최댓값을 갖지 않는 상자들을 제거
     - 과정: 확률이 가장 높은 상자를 취함 -> 각 상자의 IOU 계산 - > 특정 입계값을 넘는 상자 제거
       - 가장 좋은 것만 남김
- 모델 성능평가
  - 정밀도(precision) & 재현율(recall)
    - TP(예측이 동일 클래스의 실제 상자와 일치), FP(예측이 실제 상자와 일치하지 않는지), FN(실제 분류값이 그와 일치하는 예측을 갖지 못함)
    - precision = TP/TP+FP
    - recall = TP/TP+FN
  - 정밀도-재현율 곡선(Precision-Recall Curve)
    - 0~1 사이로 신뢰도 출력
    - 임계값 T, T=1: 정밀도 높고 재현율 낮음 신뢰도가 높은 예측만 유지, T=0: 정밀도는 낮음 재현율 높음 FP가 많아짐
    - EX. 어떤 보행자든 놓치면 안됨 -> 재현율 높힘
    - EX. 주식시장은 일부 놓쳐도 되지만, 잘못된 기회를 피함 -> 정밀도를 높힘
    - AP(Average Precision, 평균 정밀도) 와 mAP(mean Average Precision)
      - 곡선 아래의 영역
      - 0~1사이의 값을 가짐
      - 단일 클래스에 모델 성능 정보 제공
      - 전역점수: mAP사용(각 AP의 평균)
      - mAP: 최소 2개 이상의 객체를 탐지(pascal, coco: 보통 pascal 보다 낮은 점수, 데이터 수가 많음)
  

### YOLO(2018 - ):You Only Look Once 원샷, 한번에 검출하여 빠름
- (V3)RetinaNet처럼 FPN을 도입해, 정확도를 높임
- (V4)backbone을 설계하여 사용
- (V5)pytorch 구현, 빠르다, 경량화
- 진행중...
- 256 * 256 사이즈의 이미지
- 굉장히 빠름
- 작은 크기의 물체는 탐지가 힘들

### YOLO 아키텍처
- backbone 모델 기반
- Feature Extractor(특징 추출기)
- 자체 맞춤 아키텍
- 객체 크기에 따라 small, medium, big 3개의 스케일로 나뉨
- 특정 추출기 아키텍쳐를 사용했는지 따라 성능이 다름
- 그리드크기:w*h, 특징 볼륨깊이: D(depth)

### YOLO 계층 출력
- 마지막 층 출력: x * h * M
  - M = B(C+5) 
    - B: 그리드 셀당 경계 상자 개수
    - C: 클래스의 개수
    - 5를 더한 이유: 해당 값 만큼의 숫자 예측 (+5만큼 더 예측)
      - 경계 상자의 중심 좌표 + 경계상자의 너비, 높이
      - c: 객체가 경계 상자에 있다는 신뢰도

### 앵커 박스(Anchor Box)
- V2도입
- 사전에 정의된 박스
- 객체에 가장 근접한 앵커 박스로 맞춤(refine)
1. 앵커박스 grid
2. bonding box + confidence: 확률적으로 높은 박스 
3. Class probablity map: 클래스의 객체를 분류
4. 2+3을 합쳐, bounding box를 찾고 객체 클래스를 찾음


### 코드 실행 순서
1. 데이터셋 다운로드 (git clone)
2. 모델 구성 
3. 학습 (train)
4. 검증 (val-test)
5. 추론 (test Inference)
6. 모델 내보내기(weight)