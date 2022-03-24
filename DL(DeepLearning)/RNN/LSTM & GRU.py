#순환신경망의 셀: 이전 메모기를 기억하고 있음 = 메모리셀


### LSTM 셀(long short term memory): 단기기억을 길게함
#sigmoid 함수 적용 * 이전상태h + tanh함수 적용
#셀 자체 내에서 활용되는 상태값


### LSTM 신경망
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8)) #8개의 뉴런개수, 순환되는 은닉쉘,
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

### 드롭아웃: 과대적합 규제
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))   #순환신경망 드롭아웃
model2.add(keras.layers.Dense(1, activation='sigmoid'))


