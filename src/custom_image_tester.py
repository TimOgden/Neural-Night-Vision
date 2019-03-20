import cv2
import matplotlib.pyplot as plt
import numpy as np
from paper_cnn import Paper_CNN
import time
from cv2 import VideoCapture

def load_image(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (1616,1080))
	model = Paper_CNN(1080,1616,3, 'paper_model')
	model.model = model.build_model()
	model.load_model('./weights/reg_model_best.h5')

	plt.subplot(1,2,1)
	plt.imshow(img)
	plt.title('Original')

	img = np.reshape(img, (1, 1080, 1616, 3))/255.
	yhat = np.reshape(model.predict(img), (1080,1616,3))
	print((np.min(yhat),np.max(yhat)))
	yhat = np.clip(yhat, 0, 255)
	plt.subplot(1,2,2)
	plt.imshow(yhat)
	plt.title('CNN Prediction')
	plt.show()

def take_pic():
	cam = VideoCapture(0)
	s, img = cam.read()
	if s:
		img = cv2.resize(img, (1616,1080))
		model = Paper_CNN(1080,1616,3, 'paper_model')
		model.model = model.build_model()
		model.load_model('./weights/reg_model_best.h5')

		plt.subplot(1,2,1)
		plt.imshow(img)
		plt.title('Original')

		img = np.reshape(img, (1, 1080, 1616, 3))/255.
		yhat = np.reshape(model.predict(img), (1080,1616,3))
		print((np.min(yhat),np.max(yhat)))
		yhat = np.clip(yhat, 0, 255)
		plt.subplot(1,2,2)
		plt.imshow(yhat)
		plt.title('CNN Prediction')
		plt.show()
	else:
		print('Couldn\'t take picture!')

if __name__=='__main__':
	#load_image('../Sony/test.jpg')
	take_pic()