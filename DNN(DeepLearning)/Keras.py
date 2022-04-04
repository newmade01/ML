from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)    #6만개 샘플,28*28픽셀
print(test_input.shape, test_target.shape)
#fig: 전체 그래프를 저장하는 변수, axes: 그 안의 subplots을 저장하는 변수
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
#plt.show()
print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28) #3차원 ->1차원 (샘플수, 높이*너비)

#Keras 모델: 층을 만듬
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)
#Dense:  keras의 기본층 ,(밀집층 = 완전연결층 =fully connected layer)
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,)) #(출력층 10개의 뉴런= 클래스의 갯수, 활성화함수, 샘플의 크기)
model = keras.Sequential(dense) #Sequential: 순서대로 넣음

#모델 설정
model.complie(loss='sparse_categorical_crossentropy', metrics='accuracy')   #손실함: 1.이진분류: loss='binary_crossentropy', 2.다중분류: loss='categorical_crossentropy', sparse: 정수 타깃의 원핫 인코딩
print(train_target[:10])

model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)  #검증셋트 평가