from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from utility import train_test_split
from numpy import unique
from utility import load_data, one_hot
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the image',
    default='grey_images_25')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--sample_percent', type=float, default=1.0)

args = parser.parse_args()
image_dir = args.image_dir
batch_size = args.batch_size
epochs = args.epochs
save_name = args.save_name
sample_percent = args.sample_percent

X, labels = load_data(image_dir, sample_percent=sample_percent)
length, width = X[0].shape
X = X.reshape(-1, length, width, 1)
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
model.add(Flatten(input_shape=(length, width, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_unique_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print(score)
if save_name is not None:
    model.save(save_name + "-model")
    model.save_weights(save_name + "-weights")
