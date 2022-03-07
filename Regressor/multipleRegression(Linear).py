#-*- coding:utf-8 -*-
#다중회귀: 특성이 많아질수록 성능이 올라감

#데이터준비
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy() #특성을 나타내기 쉽게 to_numpy 사용
print(perch_full)

poly = PolynomialFeatures()
poly.fit([[2,3]]) #특성을 입력, 조합을 만들어내는 정도의 역할 , 학습X
poly.transform([[2,3]])

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

print(train_poly.shape) #행렬의 갯수 = 샘플 * 특성

poly.get_feature_name() #특성의 이름 출력

test_poly = poly.transform(test_input)

#LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)

lr.score(train_poly, train_target)
lr.score(test_poly, test_target)

#Degree를 추가하여 더 많은 특성 추가하기
poly =PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) #degree의 갯수가 늘어나, 특성의 갯수가 매우 많아짐
lr.fit(train_poly, train_target)
lr.score(train_poly,train_target)
lr.score(test_poly, test_target)
#결과: 특성의 갯수 > 샘플 수  => 복잡한 알고리즘 => train.score가 완벽에 가까워짐 => 매우 큰 과대적합 => 규제가 필요(부드러운 일반화된 모델, 가중치(기울기)를 작제 만듬)

#규제점 기울기를 줄이기 위한 표준화 = 특성의 scale을 조졍
from sklearn.preprocessiong import StandardScaler #평균&표준편차 사용하여 반드시 test도 변환이 필요!!

ss = StandardScaler()
ss.fit(train_poly) #평균&표준편차 구하는 역

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#규제1: 릿지회귀 =L2규제 = (가중치)**2
from sklearn.linear_model import Ridge #alpha 매개변수 (default: alpha=1) 보통 10의 배수 사용 , 적절한 규제 강도는 test.score와 train.score의 가까운 지점
ridge = Ridge()
ridge.fit(train_scaled, train_target)
ridge.score(test_scaled, test_target)

#규제2: 라쏘회구 = L1규제 =  가중치의 절댓값 (특성을 사용하지않는 경우도 발생)
from sklearn.linear_model import Lasso#alpha 매개변수 (default: alpha=1)
lasso = Lasso()
lasso.fit(train_scaled, train_target)
lasso.score(test_scaled, test_target)

###############일반적으로 L2규제 릿지회귀 사용