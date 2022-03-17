import matplotlib.pyplot as plt
import tensorflow as tf
# Check for TensorFlow GPU access
print(tf.config.list_physical_devices())
# See TensorFlow version
print(tf.__version__)

from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)    #6만개 샘플,28*28픽셀
print(test_input.shape, test_target.shape)

#fig: 전체 그래프를 저장하는 변수, axes: 그 안의 subplots을 저장하는 변수
fig, axs = plt.subplots(1, 10, figsize=(10, 10))

for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')

plt.show()