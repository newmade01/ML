from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model_ebd = Sequential() #RNN

model_ebd.add(Embedding(1000, 32)) #입력, 출력
model_ebd.add(SimpleRNN(8))
model_ebd.add(Dense(1, activation='Sigmoid'))

model_ebd.summary()