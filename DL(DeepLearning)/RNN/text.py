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
def forpass(self, x):
    for x in seq:
        z1 = np.dot(x, self.wx1) + np.dot(self.h[-1], self.w1h) +self.b1
        h = np.tanh(x) #활성화 함수
        self.h.append(h) #이전의 h값을 계속 사용하기 떄문에 버리면 안됨
        z2 = np.dot(h, self.w2)  + self.b2 #출력층
    return z2

#역방향 계산
def backprop(self, x, err):
    m = len(x) #sampel num

    #w와 b에 대한 그레이디언트
    w2_grad = np.dot(self.h[-1].T, err) / m
    b2_grad = np.sum(err) / m
    #배치차원 & 타임스텝 차원 변경
    seq = np.swapaxes(x, 0, 1)

    w1h_grad = w1x_grad =b1_grad =0
    err_to_cell = np.dot(err, self.w2.T) * (1 - self.h[-1] **2) #가장 마지막 원소가 최종값

    #모든 타임스텝을 for문으로 돌아 그레이디언트 전파
    for x, h in zip(seq[::-1][:10], self.h[:-1][::-1][:10]):
        w1h_grad += np.dot(h.T, err_to_cell) #hp의 값
        w1x_grad += np.dot(x.T, err_to_cell)
        b1_grad += np.sum(err_to_cell, axis=0)

        err_to_cell = np.dot(err_to_cell, self.w1h) *(1-h**2)

    return w1h_grad, w1x_grad, b1_grad, w2_grad, b2_grad


#모델 훈련
rn = RecurrentNetwork(n_cells=32, batch_size=32, learning_rate=0.01) #n_cell: unit, 뉴런, 순환층

rn.fit(x_train_onehot, y_train, epochs=20, x_val=x_val_onehot, y_val=y_val)


