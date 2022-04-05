'''
SGD 확률적 경사 하강법
-데이터 전처리
-데이터셋 나누기
-특성_scaled(데이터를 표준화) , SGD
-최적화, SGDClassifier : 정확도 출력
-과대적합 -> 조기종료
'''


#데이터전처리
import numpy as np
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

#학습, 검증 셋 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

#특성의 스케일을 조, 경사하강법
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

print(train_scaled)
print (test_scaled)

#최적화
#확률적 경사하강법 알고리즘SGDClassifier(<-> SGERegressor):정확도 출력
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42) #logistic 함수, max_iter=epoch와 동일, random_state: 균일하게
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #확률적 경사하강법: 하나씩 샘플을 넣어서 계산
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)#partial_fit: 기존에 W, B를 유지해서 한번 더 학습
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

#epoch가 늘어남 => 과대적합 =>테스트 값이 작아짐 => 조기종료 필요

#조기종료
sc =SGDClassifier(loss='log', random_state=42)
train_score=[]
test_score=[]

classes =np.unique(train_target) #partial_fit에서는 클래스의 갯수를 따로 저장필요
for _ in range(0, 300):
    sc.partial_fit(train_scaled. train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
sc = SGDClassifier(loss = 'log', max_iter=100, tol=None, random_state=42) #max_iter=100 , 에포크 최적점
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

