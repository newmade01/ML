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