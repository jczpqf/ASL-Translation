from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from numpy import unique
from utility import load_data, one_hot
import numpy as np

X, labels = load_data('./grey_images')
X = X.reshape(-1, 25, 25, 1)
X = X.astype('float32')
X /= 255
unique_y = unique(labels)
num_unique_labels = len(unique_y)
mapping = {label: one_hot(i, num_unique_labels)
           for i, label in enumerate(unique_y)}
y = np.array([mapping[label] for label in labels])
y = y.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = Sequential()


model.add(Flatten(input_shape=(25, 25, 1)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_unique_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64, epochs=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print(score)
