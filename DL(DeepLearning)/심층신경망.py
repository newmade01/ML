###입력층(이미지가로*세로) -> 은닉층(활성화 함수) -> 출력층 -> softmax, sigmiod
###활성화 함수: sigmoid, relu, tanh(하이퍼블릭탄젠트)

import tensorflow as tf
import keras.models import Sequential
import keras.layers import Dense, Activation

###은닉층(hidden Layer) -> 출력층
dense1 = keras.layer.Dense(100, activation='sigmoid', input_shape=(784,))#은닉층
dense2 = keras.layers.Dense(10, activation='softmax')    #출력층
model = keras.Sequential([dense1, dense2])

###summary()
model.summary()
#fit(batch_size=32) + 미니배치: 기본값
#output shape => (샘플의 갯수, 뉴런수)
#Param => 가중치+절편+모델Params의 갯수, 입력층뉴런*은닉층뉴런+은닉층뉴런

###층을 추가하는 다른 방법1
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784, ), name='hidden')
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 mnist model')

###층을 추가하는 다른 방법2
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784, ), name='hidden'))
model.add(keras.layers.Dense(10, activation='softmax', name='output'))

###모델훈련
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

#sigmoid 함수의 단점: 너무 크거나 너무 작으면 변화된 값이 거의 없(포화)
###relu 함수(=max) & Flatten층
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))    #Flatten층: 라이브러리 구성상 편하게 만들어줌, 1차원배열 =>2차원배열 펼쳐줌
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary() #flatten은 학습되는 params가 없음

###옵티마이저(optimizer) = hyper parameter
#기본 경사 하강법      :   SGD -> 모멘텀 -> 네스테로프모멘텀
#적응적 학습률 옵티마이저:           모멘텀, RMSProp-> Adam
model.compile(optimizer='sgd')  #확률적 경사하강법(미니배치=32)
sgd = keras.optimizers.SGD(learning_rate=0.1)
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True) #모멘텀 , 네스테로프모멘텀
model = keras.Sequential()
model.compile(optimizer='adam')    #Adam 지정
model.fit(train_scaled, train_target, epochs=5)     #fit() 이후에도 추가로 모델을 만들수 있음
model.evaluate(val_scaled, val_target)


###앙상블 랜덤포레스트 vs. 심층 신경망 (표현학습: 특징을 잘 찾음)
#각각 개별 계산 후 합침, 순차적 훈련 vs. 한꺼번에 동시에 계산
#다른 층 쌓을 수 X vs. 여러 층을 쌓아서 단계별로 학습 가능
#DB, 엑셀, 정형화된  vs. 이미지, 텍스트, 비정형화된
