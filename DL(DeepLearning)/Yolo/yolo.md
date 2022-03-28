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