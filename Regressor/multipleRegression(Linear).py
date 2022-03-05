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
#결과: 특성의 갯수 > 샘플 수  => 복잡한 알고리즘 => train.score가 완벽에 가까워짐 => 매우 큰 과대적합