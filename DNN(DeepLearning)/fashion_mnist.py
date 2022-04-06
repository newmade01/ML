import matplotlib.pyplot as plt
import tensorflow as tf
# Check for TensorFlow GPU access
print(tf.config.list_physical_devices())
# See TensorFlow version
print(tf.__version__)

from tensorflow import keras
import numpy as np
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)    #6만개 샘플,28*28픽셀
print(test_input.shape, test_target.shape)

#fig: 전체 그래프를 저장하는 변수, axes: 그 안의 subplots을 저장하는 변수
fig, axs = plt.subplots(1, 10, figsize=(10, 10)) #nrows=1, ncols=10, (index)
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')

#plt.show()

print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))

#로지스틱 회귀: 각 필셀마다 다른 가중치로 클래스를 곱함
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28) #3차원 ->1차원 (샘플수, 높이*너비)
print(train_scaled.shape)   #(샘플수, 2차원*3차원값)
sc = SGDClassifier(loss='log', max_iter=5, random_state=42) #max_iter =  epoch 반복횟수 #2진분류: 손실함수+sigmoid / 다중분류: 10개의 이진분류 사용+softmax
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1) #기본값 5fold # n_jobs 옵션 사용하면 CPU 코어갯수-> 병렬화 옵션 사용 X 
print(np.mean(scores['test_score']))

