import tensorflow as tf

def training(self, x, y):
    m = len(x) #image num

    with tf.GradientTap() as tape:
        z = self.forpass(x)#정방향 계산

        #loss 값 logit을 대상으로 손실값을 계산
        loss = tf.nn. softmax_cross_entropy_with_logits(y, z)
        loss = tf.reduce_mean(loss) #손실 평균화