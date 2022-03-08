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

#로지스틱회귀 이진분류
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

#sigmoid 함수 계산
from scipy.special import expit
print(expit(decision))

