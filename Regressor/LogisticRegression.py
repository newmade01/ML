import numpy as np
import pandas as pd

#데이터 준비
fish =  pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())  #fish.head() #테이블 형태로 출력
fish_input = fish[['Weight', 'Length','Diagonal', 'Height', 'Width']].to_numpy() #열특성 넘파이 행변환해서 넣음
fish_target = fish['Species'].to_numpy()

#K-최근접이웃 다중분류
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3) #기본값 n_neighbors=5
kn.fit(train_scaled, trian_target) #문자열 자동으로 정수로 바꿔서 계산
print(kn.classes_)  #생선의 속성 확인 , _: 타겟으로부터 추출한 값
kn.predict(test_scaled[:5])
proba =kn.predict_proba(test_scaled[:5])    #각 확률을 나타냄
np.round(proba,decimals=4) #deciamal: 반올림할 위치

#로지스틱 회귀: 분류알고리즘
#sigmoid 함수: 0~1까지 변경 OR Logistic 함수(tanh 도 종종 사용됨) 0.5가 기준이됨(0.5=음성클래스)
#skearn.predict(): Z값으로 판단, predict_proba(): 파이값으로 판단

#로지스틱회귀: 이진분류
bream_smelt_index = (train_target == 'beam') | (train_target == 'smelt')    #boolean index
train_bream_smelt = train_scaled[bream_smelt_index]
target_bream_smelt = train_target[bream_smelt_index]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)   #이름은 알파벳 순서대로 할당
lr.predict(train_bream_smelt[:5])   #분류된 이름
lr.predict_proba(train_bream_smelt[:5]) #확률
print(lr.coef_, lr.intercept_) #.coef_: 곱해야하는 값, 가중치, 5개의 특성 # intercept_: 절편값
#계산한 z의 값을 출력
decision = lr.decision_function(train_bream_smelt[:5])
print(decision)
#!!!!! sigmoid 함수 계산 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from scipy.special import expit
print(expit(decision))

#로지스틱회귀: 다중분류(여러개의 클래스가 있는 경우),
#OVR: 이진분류를 여러번 수행해서 다중분류를 함
lr = LogisticRegression(C=20, max_iter=1000)   #!!!!!!!!!!!!!!!!!!!1C: L2노름 규제를 기본적으로 적용 매개변수 C사용(기본값 1)(올라가면 규제 약=>LinearRegression과 반대) ,max_iter: 반복횟수(기본값 100)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))  #확률출력
print(lr.coef_.shape, lr.intercept_.shape)  #lr.coef_.shape: (클래스 갯수 ,특성과 곱해지는 계수)
#!!!!!!!!!! softMax함수 계산    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#softMax = e의 z승 값/sum(e의 z승값) =  총 더한 값을 1로 만들어줌
decision = lr.decision_function(test_scaled[:5])    #Z값 출력, 5개의 샘플
print(np.round(decision, decimals=2))
import scipy.special import softmax
proba = softmax(decision, axios=1)
print(np.round(proba, decimals=3))


#결론,
#1. decision_function: 선형함수의 양성, 음성을 찾아냄(=>predicat)
#2. predict_proba: 확률의 값들을 찾아냄
#3. Sigmod 또는 sotfmax사용