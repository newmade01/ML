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
  

### ㅣㅣ