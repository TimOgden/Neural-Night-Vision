import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from dark_image_cnn import Dark_Image_CNN
from larger_dark_image_cnn import Large_Dark_Image_CNN
from paper_cnn import Paper_CNN
from keras.models import load_model
import time

width = 128 #Should be 1616
height = 128 #Should be 1080
def show_img(img):
	plt.imshow(img)
	plt.show()

def reshape_img(img):
	#img = img / 255.
	return np.reshape(img, (-1, img.shape[0], img.shape[1], 3))

def load_norm_img(file):
	return cv2.resize(cv2.imread(file), (width,height))

def load_gray(file):
	return cv2.resize(cv2.imread(file, 0), (width,height))

def clean_up_prediction(img):
	return np.reshape(img, (width,height,3))

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
	img_x = cv2.resize(cv2.imread(x_train), (width,height)) / 255.
	img_x = np.reshape(img_x, (-1,img_x.shape[0],img_x.shape[1],img_x.shape[2]))
	return img_x

def compare_imgs(im_num):
	short_filename = '../screenshots/short/short/short-{}.jpg'.format(im_num)
	long_filename = '../screenshots/long/long/long-{}.jpg'.format(im_num)
	#short_filename = '../myimages/short_001.jpg'
	#long_filename = '../myimages/long_001.jpg'
	model = None
	with tf.device('/cpu:0'):
		model = Paper_CNN(width,height,3, 'paper_model')
		model.model = model.build_small_model()
		model.load_model('./weights/working_small_model_best.h5')
	original_image = load_norm_img(short_filename)
	gray_original = load_gray(short_filename)
	desired_image = load_norm_img(long_filename)
	img = load_norm_img(short_filename)
	
	hist_time = time.time()
	hist = cv2.equalizeHist(gray_original)
	hist_time = time.time() - hist_time
	model_time = time.time()
	y_hat = model.predict(reshape_img(original_image))[0]
	y_hat = clean_up_prediction(y_hat)
	print(y_hat.shape)
	model_time = time.time() - model_time
	
	
	#y_hat = model.model.predict_generator(generate_arrays_from_file('../new_test.txt'), steps=50, verbose=1)[0] * 255.
	plt.subplot(1,4,1)
	plt.title('Short Exposure')
	plt.imshow(convert_to_rgb(original_image))
	#plt.imshow(original_image)
	plt.subplot(1,4,2)
	plt.title('Long Exposure')
	plt.imshow(convert_to_rgb(desired_image))
	#plt.imshow(desired_image)
	plt.subplot(1,4,3)
	plt.title('Histogram Equalizer')
	plt.imshow(hist, cmap='gray')
	plt.subplot(1,4,4)
	plt.title('Prediction')
	print('Time for histogram balancing:',hist_time)
	print('Time for model prediction:', model_time)
	plt.imshow(np.clip(y_hat, 0, 256)/255.)
	plt.show()

def get_maxs():
	with open('../val.txt', 'r') as f:
		maxs = {}
		for line in f:
			space = line.index(' ')
			x_train = line[:space].strip()
			y_train = line[space+1:].strip()
			img_x = cv2.imread(x_train)
			maximum = np.max(img_x)
			if int(maximum) in maxs:
				maxs[int(maximum)] += 1
			else:
				maxs[int(maximum)] = 1
		print(maxs)
		plt.bar(list(maxs.keys()),maxs.values())
		plt.show()

if __name__=='__main__':
	for i in range (5):
		compare_imgs(i*4)
	
			
