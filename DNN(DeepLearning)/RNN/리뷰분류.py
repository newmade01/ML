#리뷰가 긍정 or 부정 예측(감성분석)
#NLP, 말뭉치, 토큰, 어휘 사전(토큰의 집합)

#데이터 불러오기
from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
print(train_input.shape, test_input.shape)

print(train_input[0])   #문장의 시작 1, 문장에 포함되지 않은 단어 비움 대신 2,
print(train_target[:20])    #긍정 1, 부정 0

###훈련세트 준비
import sklearn.model_selection from train_test_split

trian_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

lengths = np.array([len(x) for x in train_input])   #train 샘플들을 모두 순회
print(np.mean(lenghts, np.median(lenghts)))
#np.mean(lengths):단어의 갯수, np.median: 중간값
#HyperParameter적당한 길이의 토큰 길이 선정 필요

plt.hist(lengths)   #문장의 길이
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

###시퀀스 패딩: 남는 빈 공간을 0으로 채워주는 작업 진행
import tensorflow.kearas.preprocessing.sequence import pad_sequences

trian_seq = pad_sequences(train_input, maxlen=100) #maxlen: 절적한 단어의 최대길이, 미디움값 보다는 작게 설정, 기본 문장보다 길면 앞부분을 자름(0, =문장의 뒷부분이 조금 더 의미를 가짐)
print(trian_seq.shape)  #(20000, 100): (샘플개수, 토큰(타임스텝)개수) 100개까지 numpy 배열로
print(trian_seq[0]) #
print(trian_seq[0][-10:])   #마지막 10개 출력해서 0으로 패딩된 부분을 확인

###순환 신경망(simple RNN) 모델 만들기
from tensorflow import keras
model = keras.Sequence()
model.add(kearas.layers.SimpleRNN(8, input_shape=(100, 500))) #(뉴런의 갯수, inputshpe: (토큰개수(타임스텝길이), 입력값500?원핫인코딩))
model.add(kearas.layers.Dense(1, activation='sigmoid')) #이진분류: Dense (뉴런개수, 활성화함수)
#RNN층 -> Dense층 순서는 Flatten이 필요 없음!!!!!!

###원-핫 인코딩
trian_oh = keras.utils.to_categorical(train_seq)
print(trian_oh.shape) #(20000, 100, 500): 500은 최대의 값을 찾아ㅜㅁ
print(trian_oh[0][0][:12])  #첫번쩨 샘플에 첫번쩨 토큰 원핫인코딩
print(np.sum(trian_oh[0][0]))   #하나의 위치만 1, 나머지 0 => 1값 출력

val_oh = keras.utils.to_categorical(val_seq)

model.summary()
#순환된 은닉상태 h는 뉴런의 갯수로 dense층의 output값
#순환층에서 완전연결

###모델 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)  #10의 -4승
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_cb = kearas.callbacks.ModelCheckpoint('best-simplernn-model.h5')
early_stopping_cb = kearas.callbacks.EarlyStopping(patience=3, restore_best_weights=True)   #

history = model.fit(trian_oh, train_target, epochs=100, batch_size=64, validation_data=(val_oh, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

###임베딩!!!!!!: 원핫인코딩은 0,1로 극단적으로 되어있어 서로 관계를 알 수 없음, 실수 백터로 변환, 비슷한 관계를 알 수 있음
#두번째 모델
model2 = keras.Sequential()

model2.add(keras.layers.Embedding(500, 16, input_length=100))   #(입력차원, 출력차원, 타임스텝개수): 처리하는 백터의 개수가 줄어듬
model2.add(keras.layers.SimpleRNN(8))
model2.add(kearas.layers.Dense(1, activation='sigmoid'))
model.summary()
#임베딩층에 의해 뉴런의 개수가 많이 줄었지만, 정확도는 떨어지지 않음


