import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from dark_image_cnn import Dark_Image_CNN
from larger_dark_image_cnn import Large_Dark_Image_CNN
from paper_cnn import Paper_CNN
from keras.models import load_model
import time

def reshape_img(img):
	img = img / 255.
	return np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2]))

def load_norm_img(file):
	return cv2.resize(cv2.imread(file), (1616,1080))

def load_gray(file):
	return cv2.resize(cv2.imread(file, 0), (1616,1080))

def clean_up_prediction(img):
	return np.reshape(img, (1080,1616,3)) * 255.
def convert_to_rgb(bgr_img):
	b,g,r = cv2.split(bgr_img)       # get b,g,r
	return cv2.merge([r,g,b])     # switch it to rgb

def generate_arrays_from_file(path):
	while True:
		with open(path) as f:
			for line in f:
				# create numpy arrays of input data
				# and labels, from each line in the file
				x1 = process_line(line)
				yield ({'up_sampling2d_1_input': x1})

def process_line(line):
	space = line.index(' ')
	x_train = line[:space].strip()
	img_x = cv2.resize(cv2.imread(x_train), (1616,1080)) / 255.
	img_x = np.reshape(img_x, (-1,img_x.shape[0],img_x.shape[1],img_x.shape[2]))
	return img_x


if __name__=='__main__':
	short_filename = '../Sony/short/10003_06_0.1s.jpg'
	long_filename = '../Sony/long/10003_00_10s.jpg'
	model = None
	with tf.device('/cpu:0'):
		model = Paper_CNN(1080,1616,3, 'paper_model')
		model.model = model.build_model()
		model.load_model('./weights/full_layers_chkpt_0020.h5')
	original_image = load_norm_img(short_filename)
	gray_original = load_gray(short_filename)
	desired_image = load_norm_img(long_filename)
	img = load_norm_img(short_filename)
	model_time = time.time()
	y_hat = model.predict(reshape_img(img))[0]
	y_hat = clean_up_prediction(y_hat)
	model_time = time.time() - model_time
	hist_time = time.time()
	hist = cv2.equalizeHist(gray_original)
	hist_time = time.time() - hist_time
	
	
	#y_hat = model.model.predict_generator(generate_arrays_from_file('../new_test.txt'), steps=50, verbose=1)[0] * 255.
	plt.subplot(1,4,1)
	plt.imshow(convert_to_rgb(original_image))
	plt.subplot(1,4,2)
	plt.imshow(convert_to_rgb(desired_image))
	plt.subplot(1,4,3)
	plt.imshow(hist, cmap='gray')
	plt.subplot(1,4,4)
	print('Time for histogram balancing:',hist_time)
	print('Time for model prediction:', model_time)
	plt.imshow(convert_to_rgb(y_hat))

	plt.show()
