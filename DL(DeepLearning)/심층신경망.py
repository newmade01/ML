#입력층(이미지가로*세로) -> 은닉층(활성화 함수) -> 출력층 -> softmax, sigmiod
#활성화 함수: sigmoid, relu, tanh(하이퍼블릭탄젠트)

import tensorflow as tf
import keras.models import Sequential
import keras.layers import Dense, Activation
#은닉층(hidden Layer) -> 출력층
dense1 = keras.layer.Dense(100, activation='sigmoid', input_shape=(784,))#은닉층
dense2 = keras.layers.Dense(10, activation='softmax')    #출력층
model = keras.Sequential([dense1, dense2])

#summary()
model.summary()
#fit(batch_size=32) + 미니배치: 기본값
#output shape => (샘플의 갯수, 뉴런수)
#Param => 가중치+절편+모델Params의 갯수, 입력층뉴런*은닉층뉴런+은닉층뉴런

#층을 추가하는 다른 방법1
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784, ), name='hidden')
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 mnist model')

#층을 추가하는 다른 방법2
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784, ), name='hidden'))
model.add(keras.layers.Dense(10, activation='softmax', name='output'))

#모델훈련
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

#relu 함수 & Flatten층
model = keras.Sequential()
model.add()
model.add()
model.add()
model.summary()