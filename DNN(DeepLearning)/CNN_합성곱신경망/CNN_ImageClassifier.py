from tensorflow import keras
from sklearn.model_selection import train_test_split

###로드
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaeld = train_input.reshape(-1, 28, 28, 1)/255.0 #(샘플수, 픽셀, 픽셀, 차원) #이미지, 2D 이미지 그대도 사용, 3차원으로 사용 #-1값: 처음 갯수 그대로 사용
train_scaeld, val_scaled, train_target, val_target = train_test_split(train_scaeld, train_target, test_size=0.2, random_state=42)

###첫번째 합성곱층: 적은 갯수의 params로도 이미지의 특징을 잘 잡아냄
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))   #배치차원은 써주지 않음
model.add(keras.layers.MaxPooling2D(2)) #최대풀링(2*2) 정방향으로 풀링, 반으로 축소
###두번째 합성곱층 추가+완전 연결층(Dense)
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))   #채널의 갯수 64로 늘림
model.add(keras.layers.MaxPooling2D(2)) #(2*2) pooling
#분류: 1차원 배열로 총 펼침 입력값
model.add(keras.layers.Flatten())   #입력배열을 1차원으로 펼침
model.add(keras.layers.Densc(100, activation='relu'))   #100개의 뉴런 은닉층, 가중치params가 너무 많아져 과대적합 될 수 있음
model.add(keras.layers.Dropout(0.4))    #과대적합 규제
model.add(keras.layers.Dense(10, activation='softmax')) #10개의 뉴런 출력층

model.summary()

###plot_model()
keras.utils.plot_model(model, show_shpes=True)  #각 층의 연결 구성 흐름

###complie & train학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')    #모델 가중치 저장
early_stopping_cb =keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True) #떨어지는 지점에서 2번 건너뛰고 , 최적의 순간으로 돌아감
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

###평가(validation) & 예측(predict)
#predict()는 테스트 이미지의 분류 결과를 예측합니다. 반환값이 예측 확률입니다.
#evaluate()는 테스트 이미지 데이터 세트를 입력해서 성능 평가를 합니다. 반환값이 정확도등의 Metric입니다.
model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

predi = model.predict(val_scaled[0:1])  #첫번째 이미지의 예측한 10개의 확율

###테스트 세트 점수(testSet)
test_scaled = test_input.reshape(-1, 28, 28, 1) /255.0
model.evaluate(test_scaled, test_target)
