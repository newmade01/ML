#선형회귀 (직선의 방정식)
#문제점: 무게가 음수로 낮아질 수 있음, 현실적으로 쓰기엔 애매

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
