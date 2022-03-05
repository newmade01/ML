#선형회귀 (직선의 방정식)
#문제점: 무게가 음수로 낮아질 수 있음, 현실적으로 쓰기엔 애매

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


lr=LinearRegression()
lr.fit(train_input,train_target)
#예측, 직서의 방정식 활용 y=aX+b
lr.predict([[50]])

#기울기: .coef_, 절편: intercept_
print(lr.coef_, lr.intercept_)

plt.scatter(train_input, train_target)

#1차 방정식 그래프 그리기 (x값, y값) 15~ 50까지의 값
plt.plot([15, 50],[15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
#그래피 마킹
plt.scatter(50, 1241.8, marker='^')
plt.show()

lr.score(train_input, train_target)
lr.score(test_input, test_target)


#다항회귀 추가 X^2, X의 값을 열로 붙침
train_poly = np.column_stack((train_input **2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
lr.fit(train_poly, train_target)
lr.predict([[50**2, 50]]) #임의의 값 예
print(lr.coef_, lr.intercept_)
#구간별 직선 임의 선정 15~49까지
point = np.arrange(15, 50)
plt.scatter(train_poly, train_target)#훈련 세트 산점도
#2차 방정식 그래프 그리기
plt.plot(point, lr.coef_[0]*point**2+lr.coef[1]*point+lr.intercept_)
#50센치 농어 데이터 그리기
plt.scatter([50],[1574],marker='^')
plt.show()
lr.score(train_poly, train_target)
lr.score(test_poly, test_target)

