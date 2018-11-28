from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utility import load_data
import argparse


parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the image',
    default='grey_images_25')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--sample_percent', type=float, default=1.0)
parser.add_argument('--classifier_type', type=str, default='rf',
                    choices=['rf'])
parser.add_argument('--classifier_args', type=dict, default={})

args = parser.parse_args()
image_dir = args.image_dir
save_name = args.save_name
sample_percent = args.sample_percent
classifier_type = args.classifier_type
classifier_args = args.classifier_args

X, labels = load_data(image_dir, sample_percent=sample_percent)
if len(X[0].shape) != 1:
    dim = 1
    for d in X[0].shape:
        dim *= d
    X = X.reshape(-1, dim)
X = X.astype('float32')
X /= 255
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1,
                                                    shuffle=False)
if classifier_type == 'rf':
    model = RandomForestClassifier(**classifier_args)

model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
