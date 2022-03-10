#결정트리 알고리즘: yes OR no
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)    #max_feature: 최대 특성의 값의 갯수
dt.fit(train_scaled, train_target)
dt.score(test_scaled, test_target)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7)) #이미지 사이즈를 키움
plot_tree(dt)
plt.show()