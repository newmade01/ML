###손실 곡선
import keras
import matplotlib.pyplot as plt

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)    #epoch를 늘리면 모델이 더 복잡해짐 #history: 객체의
print(history.history.keys())   #loss, accuracy 값이 들어있음

plt.plot(history.history['loss'])
#plt.polt(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

###검증 손실(validation_data): train, val, test의 값이 골고루
#과대적합 규제: L1, L2규제 보다는 Drop out 많이 사용
history = model.fit(train_scaled, train_target, epochs=2-, verbose=0, validation_data=(val_scaled, val_target))
print(history.history.keys())   #loss, accuracy, val_loss, val_accuracy 값이 출력

###드롭아웃: 훈련시, 은닉층 뉴런을 하나 랜덤(%, 하이퍼파라미터) 삭제, 특정 뉴런의 의존 현상 없어짐, 과대적합을 막아줌
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.3))    #30%를 끔
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

###weight 모델 저장 & 복원
model.save_weights('w.h5')  #가중치만 저장
model.load_weights('w.h5')  #불러옴

model.save('model_whole.h5')    #모델구조&가중치 모두 저장
model = keras.models.load_model('model_whole.h5')   # 모델구조&가중치 모두 불러옴

#샘플마다 가장 큰 확률 값의 index 가져옴
val_labels = np.argmax(model.predic(val_scaled), axis=-1) #예측  #axis =-1 은 axis=1 값과 동일 열..
print(np.mean(val_labels == val_target))    #True의 값들 평균이 나옴

###콜백 : 진행도중 지정한 작업을 시행
checkpoint_cb = keras.callbacks.ModelCheckpoitnt('best-model.h5')    #가장 낮은 손실값을 저장
model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])

model=  keras.models.load_model('best-model.h5')    #사용시 불러옴

###조기종료: 검증세트의 손실이 증가시
checkpoint_cb = keras.callbacks.ModelCheckpoitnt('best-models.h5')
early_stopping_cb = keras.callbacks