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

#로지스틱 회귀