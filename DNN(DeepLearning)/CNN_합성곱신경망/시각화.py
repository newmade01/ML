#22강 합성공 신경망의 시각화
import maplotlib.pyplot as plt

conv =model.layer[0] #Conv2D
print(conv.weights[0].shape, conv.weight[1].shape)
conv_weights =conv.weights[0].numpy()

plt.hist(con_weights.reshap(-1, 1)) #hist: 히스토 그램 #reshape(-1, 1): -1은 값으로 추정, 1은 1열 (행,열)
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

###이미지 시각화
fig, axs = plt.subplots(2, 16, figsize=(15,2)) #(행, 열, 사이즈)
for i in range(2):
  for j in range(16):
    axs[i, j].imshow(conv_weights[:,:, 0, i*16+j], vmin=-0.1, vmax=0.5) #[3*3원소, depth=0, 원소 32개 순환 이미지 그려줌]
    axs[i, j].axis('off')
plt.show()
#원래는 균등분포로 희미하게 나옴 

###함수형API: 활성화층 시각화
dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10, activation='softmax')

#입력층: input 값 그 자체 -> 은닉층 -> 출력(모델)
input = keras.Input(shape=...) #InputLayer클래스 객체
hidden =  dense1(intput)
output = dense2(hidden)
model  = keras.Model(input, output)

#모델 객체의 층
#!!!!!모델 객체 -  InputLayer - Conv2D - Maxpolling2D - Conv2D -  Maxpooling2D - Flatten - Dense - Dropout - dense1
#InputLayer - conv2D(=model.layers[0].output)
conv_acti = keras.Model(model.input, model.layers[0].output) #새로운 모델을 만듬

#첫번째 특성맵 시각화: 합성곱
inputs = train_input[0:1].reshape(-1, 28, 28, 1 )/ 255.0
feature_maps = conv_acti.predict(input) #훈련의 첫번째 데이터셋을 예측
print(feature_maps.shape) #(1, 28, 28, 32)

#두번째 특성맵 시각화: 합성곱
conv2_acti = keras.Model(model.input, model.layer[2].output) #두번째 활성화함수 Conv2D
feature_maps2  =conv2_acti.predict(inputs)
print(feature_maps.shape)   #(1, 14, 14, 64)
#합성곱 특성맵: 낮은층에서 저수준 학습, 단순한 모양, 패턴 , 색 (층이 깊어질수록 고수준 특성맵)
















