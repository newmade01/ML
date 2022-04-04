from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN #가장 기본적인 RNN학습

model = Sequential()

model.add(SimpleRNN(32, input_shape=(100, 100))) #32개의 유닛, 100*100개의 input 입력길이 단어
model.add(Dense(1, activation="sigmoid"))

model.summary()


#모델 훈련
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_onehot, y_train, epochs=20, batch_size=32, validation_data=(x_val_onehot, y_val))