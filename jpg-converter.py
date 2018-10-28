"""
Converts jpg images to a single channel numpy arrays
Run this file from the directory where the image folder is contained

Takes command line arguments
    The first argument is the folder where the images are contained
    The second argument is the name of the new folder to store the new images
"""

import os
import sys
import numpy as np
from PIL import Image

image_dir = 'raw_images'
new_dir = 'greyscale_images'

if len(sys.argv) >= 2:
    new_dir = sys.argv[1]
if len(sys.argv) >= 2:
    image_dir = sys.argv[0]


def remove_extension(file_name):
    """
    Removes everything at and after the first '.' character
    """
    for i, c in enumerate(file_name):
        if c == '.':
            return file_name[:i]
    return file_name


root_dir = os.getcwd()
try:
    os.mkdir(root_dir + '\\' + new_dir)
except FileExistsError:  # if directory already exists thats ok
    pass
path_labels = os.listdir(root_dir + '\\' + image_dir)
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
