from sklearn.model_selection import trian_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


train_input, test_input, train_target, test_target = train_test_split(length, weight, random_state=42)
#trian은 2차원 배열 , test는 1차원 배열 (상관없음 )

# [
#     [,],
#     [,]
# ]

# 1차원 -> 2차원 배열을 기대
train_input =train_input.reshape(-1, 1) # -1 의 의미는 나머지 차원을 할당하고 남은 차원을 알아서 할당해라
test_input = test_input.reshpe(-1, 1)

knr =  KNeighborsRegressor()
knr.fit(train_input, train_target) #훈련
knr.score(test_input, test_target) #testSet 점수 확인 (분류-정확도, 회귀-결졍계수)

#예측과 평균 값 구하기
from sklearn.metric import mean_absolute_error

test_predict = knr.predict(test_input)#예측값
mea  = mean_absolute_error(test_target, test_predict) #타깃값과 예측의 차이

#score(train) < score(test) 과소적합
#score(train) > score(test) 과대적합

#!!!!!!!!!!!!!!!!n_neighbors= 5 (default) 이웃의 갯수의 조절이 중요!!!!!!!!!!!!!!!! = 하이퍼파라미터
knr.n_neighbors = 3

#!!!!!!!!!!!!!!!!과대적합과 과소 적합의 적절한 균형점을 찾는 것이 중요!!!!!!!!!!!!!!!!!!!!!!!!!

#회귀 = 임의의 숫자를 예측하는 문제
#문제점: 극단적으로 커져도 근처에 있는 값을 활용해 예측해서 오차가 심해질 수 있음
