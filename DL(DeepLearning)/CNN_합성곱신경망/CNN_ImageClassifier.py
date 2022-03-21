from tensorflow import keras
from sklearn.model_selection import train_test_split

#로드
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaeld = train_input.reshape(-1, 28, 28, 1)/255.0 #(샘플수, 픽셀, 픽셀, 차원) #이미지, 2D 이미지 그대도 사용, 3차원으로 사용 #-1값: 처음 갯수 그대로 사용

train_scaeld, val_scaled, train_target, val_target = train_test_split(train_scaeld, train_target, test_size=0.2, random_state=42)

model = keras.Sequential()

model.add()