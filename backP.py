import tensorflow as tf

def training(self, x, y):
    m = len(x) #image num

    with tf.GradientTap() as tape:
        z = self.forpass(x)#정방향 계산

        #loss 값 logit을 대상으로 손실값을 계산
        loss = tf.nn. softmax_cross_entropy_with_logits(y, z)
        loss = tf.reduce_mean(loss) #손실 평균화

    #가중치에 대한 오차(parameter)
    weights_list = [self.conv_w, self.conv_b, self.w1, self.b1, self.w2, self.b2]

    #가중치에대한 그레디언트
    grads = tape.gradient(loss, weights_list)

    #가중치 업데이트
    self.optimizer.apply_gradients(zip(grads, weights_list))