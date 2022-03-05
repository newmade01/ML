from sklearn.model_selection import trian_test_split
import numpy as np

train_input, test_input, train_target, test_target = train_test_split(length, weight, random_state=42)
#trian은 2차원 배열 , test는 1차원 배열 (상관없음 )

# [
#     [,],
#     [,]
# ]

# 1차원 -> 2차원 배열을 기대
train_input =train_input.reshape(-1, 1) # -1 의 의미는 나머지 차원을 할당하고 남은 차원을 알아서 할당해라
test_input = test_input.reshpe(-1, 1)


