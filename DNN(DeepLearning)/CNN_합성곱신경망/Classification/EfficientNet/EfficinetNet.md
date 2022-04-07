# EfficientNet(Smller Models and Faster Traning)
- 최소한의 컴퓨팅 자원, 최대한의 성능
- 종합 선물 세트
- 모델과 데이터가 커짐, 학습 효울의 중요도 커짐
- 다른 모델들을 통해 학습 효율은 커졌지만, 파라미터 효율(모델크기)가 나빠짐
- 높은 학습 효율 & 파라미터 효율 & 높은 정확도 개선
- 파라미터 갯수가 굉장히 적어짐
- 빠른 시간안에 학습
- 
### EfficinetNet V1
- 컴퓨팅 자원에따라 , B0 ~ B7 모델 제안
- width(채널), depth(레이어), resolution(이미지 사이즈) 동시에 조절 => compound scaling method!!!! 순차적으로 scale UP
  1. Width: 채널수
  2. Depth: 레이어 개수(모델이 얼마나 깊은지)
  3. Resolution: 이미지 사이즈
- Width, Depth, Resolution 모두 독립적이지 않음, 두가지 동시에 올릴때 정확도가 높아짐 => compound scaling
- Baseline은 B0네트워크 = MobileNasNet(정확도와 속도를 NAS의 객체로 설정해 보상(Transfer)을 줌)
  - Moblie NasNet: 속도 기준 실제 디바이스에서의 latency 측정
  - EfficientNet: 범용모델로 latency 대신, Flop을 설정(Flop: 딥러닝 모델이 얼마나 빠르게 동작하는지에 대한 metric , 모바일 환경에서 돌아가는 가벼운 )
- B0를 기반으로 모델 scale Up
  1. 파이를 1로 고정한 뒤, grid search를 이용해 알파, 베타, 감마 탐색(***Grid search (격자 탐색) 은 모델 하이퍼 파라미터에 넣을 수 있는 값들을 순차적으로 입력한뒤에 가장 높은 성능을 보이는 하이퍼 파라미터들을 찾는 탐색 방법)
  2. 탐섹된 알파, 베타, 감마를 고정한 상테로, 파이를 2로 증가시켜 B1획득
  3. ***파이만 증가시켜 B2, B3,...B7 획득

### EfficientNet V2
- Efficient V1의 문제점
  1. 해상도가 높은 이미지로 학습하면 속도가 느림
      - 동일한 GPU 메모리에, 이미지가 커질경우 mini-batch size 감소 학습속도 느려짐
  2. Depthwise Seperable Convolution(일반 conv 대비 8-9배 연산량이 줄어듬, 모바일or엣지 디바이스)이 초기 layer에서 느림, 계산 방식이 아직 최적화 되지않았음(***Depth-wise Convolution은 한 번 통과하면, 하나로 병합되지 않고, (R, G, B)가 각각 Feature Map이 된다)
     - 이론적으로는 Depthwise가 빠른데, Depthwise Conv가 초기 레이어에 있을수록 실제 runtime은 더 느림(modern hardware가 효율적으로 구현하지 못함)
  3. 모든 stage를 동일하게 scaling하는 것은 sub-optimal
     - 정확도 & Flop을 최적화
     - 간단한 compound scaling rule을 이용해 모든 stage를 동일하게 scale up, 학습 속도 및 파라미터 효율을 최적화
- Efficient V1의 해결방법 => V2에 적용 방법
  1. Progressive Learning을 활용해 image 와 regularization을 동적 변경
     - 학습중, 학습 설정을 동적으로 변경
     - Regularization: RandAugment, mixup, dropout(ConvNet은 이미지 사이즈에 독립적)
       - 사이즈가 작은 이미지, 약한 regularization 필요
       - 사이즈가 큰 이미지, 강한 regularization 필요
  2. 초기 layer을 기존 MBConv 대신 (일반 conv를 사용한) Fused-MBconv로 교체 => 런타임 자체는 더 빨라짐
  3. non-uniform scaling 전략을 적용
     - 정확도 & 학습효율 & 파라미터 효율을 Nas의 objective로 설정(=>EfficientNet V2-small) 

### EfficientNet V1 vs. EfficientNet V2
  - V1: 초기 레이어에 MBConv , V2: Fused-MBConv
  - V1: 연산량이 많은 5*5 커널, V2:  전부 3*3 커널 (receptive field 줄어들어, 네트워크 후반부 여러개의 레이어를 추가)(***receptive field: 외부 자극이 전체 영향을 끼치는 것이 아니라 특정 영역에만 영향을 준다,  특정 범위를 한정해 처리를 하면 훨씬 효과적)
  - 지나치게 큰 이미지 사이즈는 메모리 & 학습 속도에 부하를 줄 수 있음 => Inference에서 최대 이미지 사이즈를 480으로 제한
  - V1: uniform한 스케일링 방식, V2: non-Uniform: Inference에서 최대 이미지 사이즈를 480으로 제한
  - V2: 저자가 layer을 추가하여 scail up, later stage에 점진적으로 추가

# EfficientNet V2
### 결과
- 정확도는 이전 알고리즘과 비슷, parameter & FLOP & Latency가 훨씬 적음
- 정확도가 어느정도 높아지면, 모델 크기를 늘려서는 더이상 정확도를 올리기 힘들다. but, 데이터 사이즈!!!를 늘리면 정확도를 올릴수 있음
- ImageNet21k로 사전 학습을 하면 충분히 효율적 (데이터 사이즈가 기본 ImageNet에 비해 10배 이상 큼)

### 이외...
- Progressive Learning을 다른 네트워크에 적용했을때도 성능이 좋아짐
- adaptive regularization 더 나은 정확도
- FixEfficientNet에서 train 입력 이미지를 test 입력 이미지보다 작게 학습하여 성능을 향상

### 결론
- multi-objective NAS, Fused-MBConv로 V1의 성능 향상
- Adaptive regularization, progressive learning 제안
- 학습 효율 & 파라미터 효율 & 정확도 향상
- 비용적으로 효율적
- 아주 강력한 정규화(regularization)

### Progressive Learning이란...
- Progressive Learning: training할 때, 이미지의 크기를 점진적으로 증가 => 학습 속도를 빠르게, but, 정확도가 감소
- 이전 모델들은 이미지 크기에 따라 모두 동인한 정규화(dropout, augmentation)적용했지만, EfficientNetdms 이미지 크기에 따라 정규화 방법을 다르게 설정하여 정확도 감소를 해결
  - 입력 이미지가 작을 때 => 약한 정규화
  - 입력 이미지가 클 때 => 오버피팅 방지위해 강한 정규화
- 적용방법: training 과저을 4 stage로 나눠 1 stage당 87epochs를 진행, stage를 지날수록 이미지 크기와 정규화 강도 높아짐

### Depthwise convolution이란...
- conv 연산량을 낮춰주어 제한된 연산량 내에 더 많은 filter를 사용
- modern accelerator를 활용하지 못해, 학습 속도를 늘게함 => Fused-MDconv사용
