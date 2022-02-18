import tensorflow as tf
import numpy as np

def init_weights(self, input_shape, n_classes):
    g = tf.initializers.glolot_uniform() #가중치 초기: 적절한 전역 초기화를 찾기위해
    self.w = tf.variable(g((3,3,1, self.n_kernels)))
    self.b = tf.variable(np.zeros(self.n_kernels), dtype=float ) #절편을 0으로 초기화 하는 것이 일반적
    n_feature = 14 * 14 * self.n_kernels #원본 28.28에서 절반으로 특성맵을 줄임
    self.w1 = tf.variable(g((n_feature,self.units))) #특성맵, 은닉층
    self.b1 = tf.variable(np.zeros(self.units) dtype=float)
    self.w2 = tf.variable(g((self.units, n_classes)))
    self.b2 =tf.variable(np.zeros(n_classes), dtype=float)

