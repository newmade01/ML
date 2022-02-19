from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

conv1 = tf.keras.sequential()
conv1.add(Conv2D(10, 3, 3 ))