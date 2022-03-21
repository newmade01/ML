#functional API
import matplotlib.pyplot as plt

conv = model.layers[0]
print(conv.weghts[0].shape, conv.weghts[1].shape)   #가중치, 절편    (픽셀, 필셀, 채널, 필터)

conv_weights = conv.weghts[0].numpy()   #일반적으로 가중치는 균등분포
plt.hist(conv_weights.reshape(-1, 1))    #1차원 열 백터로 펼침, 히스토그램 그래프
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

