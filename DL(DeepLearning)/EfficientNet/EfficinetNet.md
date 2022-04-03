#EfficientNet(Smller Models and Faster Traning)
- 최소한의 컴퓨팅 자원, 최대한의 성능
- 종합 선물 세트
- 모델과 데이터가 커짐, 학습 효울의 중요도 커짐
- 다른 모델들을 통해 학습 효율은 커졌지만, 파라미터 효율(모델크기)가 나빠짐
- 높은 학습 효율 & 파라미터 효율 & 높은 정확도 개선
- 파라미터 갯수가 굉장히 적어짐
- 빠른 시간안에 학습
- 
###EfficinetNet V1
- 컴퓨팅 자원에따라 , B0 ~ B7 모델 제안
- width(채널), depth(레이어), resolution(이미지 사이즈) 동시에 조절 => compound scaling method!!!! 순차적으로 scale UP
  1. Width: 채널수
  2. Depth: 레이어 개수(모델이 얼마나 깊은지)
  3. Resolution: 이미지 사이즈
- Width, Depth, Resolution 모두 독립적이지 않음, 두가지 동시에 올릴때 정확도가 높아짐 => compound scaling
- Baseline은 B0네트워크 = MobileNasNet(정확도와 속도를 NAS의 객체로 설정해 보상(Transfer)을 줌)
  - Moblie NasNet: 속도 기준 실제 디바이스에서의 latency 측정
  - EfficientNet: 범용모델로 latency 대신, Flop을 설정
- B0를 기반으로 모델 scale Up
  1. 파이를 1로 고정한 뒤, grid search를 이용해 알파, 베타, 감마 탐색(***Grid search (격자 탐색) 은 모델 하이퍼 파라미터에 넣을 수 있는 값들을 순차적으로 입력한뒤에 가장 높은 성능을 보이는 하이퍼 파라미터들을 찾는 탐색 방법)
  2. 탐섹된 알파, 베타, 감마를 고정한 상테로, 파이를 2로 증가시켜 B1획득
  3. ***파이만 증가시켜 B2, B3,...B7 획득

###EfficientNet V2
- Efficient V1의 문제점
  1. 해상도가 높은 이미지로 학습하면 속도가 느림
      - 동일한 GPU 메모리에, 이미지가 커질경우 mini-batch size 감소 학습속도 느려짐
  2. Depthwise Seperable Convolution이 초기 layer에서 느림, 계산 방식이 아직 최적화 되지않았음(***Depth-wise Convolution은 한 번 통과하면, 하나로 병합되지 않고, (R, G, B)가 각각 Feature Map이 된다)
  3. 모든 stage를 동일하게 scaling하는 것은 sub-optimal
      - 간단한 compound scaling rule을 이용해 모든 stage를 동일하게 scale upgkdu, 학습 속도 및 파라미터 효율을 최적화
- Efficient V1의 해결방법
  1. Progressive Learning을 활용해 image 와 regularization을 동적 변경
     - 학습중, 학습 설정을 동적으로 변경
     - Regularization: RandAugment, mixup, dropout
       - 사이즈가 작은 이미지, 약한 regularization 필요
       - 사이즈가 큰 이미지, 강한 regularization 필요
  2. 초기 layer을 MBConv 대신 Fused-MBconv로 교체
  3. non-uniform scaling 전략을 적용
