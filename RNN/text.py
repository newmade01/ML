import numpy as np
from tensorflow.keras.datasets import imdb

#data load, get
(x_train_all, y_train_all), (x_test, y_test) =imdb.load_data(skip_top=20, num_words=100)

print(x_train_all.shape, y_train_all.shape)



from tensorflow.keras.preprocessing import sequence
maxlen=100 #최대 길이를 지정
x_train_sequence = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val_sequence = sequence.pad_sequences(x_val, maxlen=maxlen)

#one_hot encoding: 자연어 벡터 형식으로 변경
from tensorflow.keras.utils import to_categorical
x_train_onehot = to_categorical(x_train_sequence)
x_val_onehot = to_categorical(x_val_sequence)

#배치 차원& 타임스텝 차원 변경: 코드를 위해 편의상
seq = np.swapaxes(x, 0, 1)

#정방향 계산 #np.dot: 행렬의 곱
for x in seq:
    z1 = np.dot(x, self.wx1) + np.dot(self.h[-1], self.w1h) +self.b1
    h = np.tanh(x) #활성화 함수
    self.h.append(h) #이전의 h값을 계속 사용하기 떄문에 버리면 안됨
    z2 = np.dot(h, self.w2)  + self.b2 #출력층
return z2