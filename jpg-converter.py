"""
Converts jpg images to a single channel numpy arrays
Run this file from the directory where the image folder is contained

Takes command line arguments
    The first argument is the folder where the images are contained
    The second argument is the name of the new folder to store the new images
"""

import os
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Parse potential arguments')
parser.add_argument(
    '--image_dir', type=str, help='the directory of the image',
    default='raw_images')
parser.add_argument(
    '--new_dir', type=str, help='the new director of the new images',
    default='grey_images')
parser.add_argument(
    '--sub_dir_labels', type=bool, default=True
)

args = parser.parse_args()

image_dir = args.image_dir
new_dir = args.new_dir
sub_dir_labels = args.sub_dir_labels


def remove_extension(file_name):
    """
    Removes everything at and after the first '.' character
    """
    for i, c in enumerate(file_name):
        if c == '.':
            return file_name[:i]
    return file_name


root_dir = os.getcwd()
path_labels = os.listdir(root_dir + '\\' + image_dir)
try:
    os.mkdir(root_dir + '\\' + new_dir)
except FileExistsError:  # if directory already exists thats ok
    pass
if sub_dir_labels:
    for label in path_labels:
        try:
            os.mkdir(root_dir + '\\' + new_dir + '\\' + label)
        except FileExistsError:  # if directory already exists thats ok
            pass
        files = os.listdir(root_dir + '\\' + image_dir + '\\' + label)
        for file in files:
            img = Image.open(root_dir + '\\' + image_dir + '\\' +
                             label + '\\' + file)  # open image
            img = img.convert('L')  # convert image to greyscale
            img = np.array(img)  # convert image to numpy array
            np.save(root_dir + '\\' + new_dir + '\\' +
                    label + '\\' + remove_extension(file), img)
else:
    files = os.listdir(root_dir + '\\' + image_dir)
    for file in files:
        # open image
        img = Image.open(root_dir + '\\' + image_dir + '\\' + file)

        img = img.convert('L')  # convert image to greyscale
        img = np.array(img)  # convert image to numpy array
        np.save(root_dir + '\\' + new_dir + '\\' + remove_extension(file), img)
