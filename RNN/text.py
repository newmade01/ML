import numpy as np
from tensorflow.keras.datasets import imdb

#data load, get
(x_train_all, y_train_all), (x_test, y_test) =imdb.load_data(skip_top=20, num_words=100)

print(x_train_all.shape, y_train_all.shape)
