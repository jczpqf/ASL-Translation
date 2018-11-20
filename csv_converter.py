import os
import argparse
import pandas as pd
import numpy as np
from math import sqrt

parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--csv_name', type=str, help='the directory of the images',
    default='raw_images')
parser.add_argument(
    '--new_dir', type=str, help='the new director of the new images',
    default='grey_images')
parser.add_argument('--new_dim', type=int, default=27)
parser.add_argument('--reshape_2d', type=bool, default=True)
parser.add_argument('--sample_percent', type=float, default=1.0)
parser.add_argument('--label_name', type=str, default='label')
parser.add_argument('--prefix', type=str, default='P')

args = parser.parse_args()

csv_name = args.csv_name
new_dir = args.new_dir
new_dim = args.new_dim
sample_percent = args.sample_percent
label_name = args.label_name
prefix = args.prefix
reshape_2d = args.reshape_2d

df = pd.read_csv(csv_name)
non_label_columns = [column for column in df.columns if column != label_name]
dim_1d = len(non_label_columns)
if reshape_2d:
    new_dim_2d = int(sqrt(dim_1d))
    if new_dim_2d * new_dim_2d != dim_1d:
        raise AssertionError('cannot reshape dimensions')

root_dir = os.getcwd()
try:
    os.mkdir(root_dir + '\\' + new_dir)
except FileExistsError:  # if directory already exists thats ok
    pass

unique_y = df[label_name].unique()
for unique in unique_y:
    try:
        os.mkdir(root_dir + '\\' + new_dir + '\\' + str(unique))
    except FileExistsError:  # if directory already exists thats ok
        pass
    single_label_df = df[df[label_name] == unique]
    X = single_label_df[non_label_columns].as_matrix()
    if reshape_2d:
        X = X.reshape(len(X), new_dim_2d, new_dim_2d)
    for i, x in enumerate(X):
        np.save(root_dir + '\\' + new_dir + '\\' +
                str(unique) + '\\' + prefix + str(i), x)
