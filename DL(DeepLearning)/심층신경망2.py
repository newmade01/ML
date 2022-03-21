###손실 곡선
import matplotlib.pyplot as plt

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)    #history: 객체의
print(history.history.keys())   #loss, accuracy 값이 들어있음

plt.plot(history.history['loss'])
#plt.polt(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

