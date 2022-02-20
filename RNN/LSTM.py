from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import Embedding

model_lstm = Sequential()
model_lstm.add(Embedding(1000, 32))
model_lstm.add(LSTM(8)) #곱하는 가중치의 종류가 많아 유닛을 줄여도 성능이 좋음
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.summary()