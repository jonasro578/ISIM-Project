# just some basic snippets which might be useful
# @author:	J. Roeper
# @date:	01.12.2019

# check https://github.com/Tejas07PSK/Melanoma-Detection for example project

# necessary steps:
# 1. Preprocessing
#	- Import images&resize
#	- Extend datasets with augmented data to improve robustness
#	- Shape detection and cropping
# 2. Feature extraction
#	- blurred average of RGB?
#	- mean greyscale value?
#	- edge detection? calculation of key points? (similar areas in the picture)


# code example for import/export of local data
# code from https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/
# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, compute color channel
	# statistics, and then update our data list
	image = Image.open(imagePath)
	features = extract_color_stats(image)
	data.append(features)
	# extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	
#_______________________________________
# function for import and scaling of pictures
# see https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8
IMG_SIZE = 300
def load_training_data():
  train_data = []
  for img in os.listdir(DIR)
    label = label_img(img)
    path = os.path.join(DIR, img)
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    train_data.append([np.array(img), label])
    # Basic Data Augmentation - Horizontal Flipping
	# TODO: Augmentation using random cropping, translations, color scale, shifts
    flip_img = Image.open(path)
    flip_img = flip_img.convert('L')
    flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    flip_img = np.array(flip_img)
    flip_img = np.fliplr(flip_img)
    train_data.append([flip_img, label])
  shuffle(train_data)
  return train_data


#_______________________________________
# extract basic image stats
# note that the feature vector in this example will be waay to simple
# code from https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/
# necessary packages
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]

	# return our set of features
	return features
	
#_______________________________________
# hu momoments for detection of image shape using OpenCV cv2 lib
# simply convert image to greyscale and define threshold
# see https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/
import cv2
def calc_hu_moments(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return(cv2.HuMoments(cv2.moments(image)).flatten())



