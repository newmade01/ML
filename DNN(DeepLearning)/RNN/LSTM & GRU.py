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

### 드롭아웃: 과대적합 규제, 훈련세트와 갭이 줄어듬, 성능 향상
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))   #순환신경망 드롭아웃,
model2.add(keras.layers.Dense(1, activation='sigmoid'))

### 2개의 층 연결 (여러개 층 추가)
#모든 타입스텝 은닉상태 출력 #기본적으로 keras는 마지막 타임스텝의 은닉 상태만 저장됨
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))    #return_sequences=True: 마지막층을 제외한 모든 타임스텝의 은닉 상태 저장
model3.add(keras.layers.LSTM(8, dropout=0.3))   #마지막 순환층
model3.add(keras.layers.Dense(1, activation='sigmoid'))
model3.summary()


### GRU 셀: LSTM의 간소화 버전, 셀 상태 X, 은닉상태와 입력만 가짐
# 이전 은닉 상태가 셀을 제어
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))
model4.summary()

