import matplotlib.pyplot as plt
import cv2
import numpy as np
from dark_image_cnn import Dark_Image_CNN

def load_image(file):
	return cv2.resize(cv2.imread(file), (1616,1080)) / 255.

def load_gray(file):
	return cv2.resize(cv2.imread(file, 0), (1616,1080))

def convert_to_rgb(bgr_img):
	b,g,r = cv2.split(bgr_img)       # get b,g,r
	return cv2.merge([r,g,b])     # switch it to rgb

if __name__=='__main__':
	model = Dark_Image_CNN(32,10) # 32 and 10 are the batch size and number of epochs, which is irrelevant in this case but whatevs

	original_image = load_image('../short/00001_00_0.1s.jpg')
	gray_original = load_gray('../short/00001_00_0.1s.jpg')
	print(gray_original.shape)
	desired_image = load_image('../long/00001_00_10s.jpg')
	hist = cv2.equalizeHist(gray_original)

	original_image_reshaped = np.reshape(original_image, (-1, original_image.shape[0], original_image.shape[1], original_image.shape[2]))
	y_hat = model.predict(original_image_reshaped) * 255.


	plt.subplot(1,4,1)
	plt.imshow(convert_to_rgb(original_image))
	plt.subplot(1,4,2)
	plt.imshow(convert_to_rgb(desired_image))
	plt.subplot(1,4,3)
	plt.imshow(hist, cmap='gray')
	plt.subplot(1,4,4)
	plt.imshow(y_hat)
	plt.show()
