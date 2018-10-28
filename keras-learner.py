from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from numpy import unique
from utility import load_data
import numpy as np

X, labels = load_data('./grey_images')
unique_y = unique(labels)
num_unique_labels = len(unique_y)
mapping = {label: i for i, label in enumerate(unique_y)}
y = np.array([mapping[label] for label in labels])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = Sequential()

model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_unique_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print(score)
