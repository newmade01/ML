#리뷰가 긍정 or 부정 예측(감성분석)
#NLP, 말뭉치, 토큰, 어휘 사전(토큰의 집합)

#데이터 불러오기
from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
print(train_input.shape, test_input.shape)

print(train_input[0])   #문장의 시작 1, 문장에 포함되지 않은 단어 비움 대신 2,
print(train_target[:20])    #긍정 1, 부정 0

#훈련 세트 준비 : 검증세트val, 조기종료
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

lengths = np.array([])