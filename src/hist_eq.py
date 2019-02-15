from obtain_images import *
import cv2
import matplotlib.pyplot as plt
import time
if __name__ == '__main__':
	images = obtain_data('../new_train.txt', amount=1, transform=shrink_greyscale_func(128, 128, 1))
	img_x = cv2.resize(cv2.imread('../short/00004_00_0.1s.jpg',0), (1616,1080))
	plt.imshow(img_x, cmap='gray')
	plt.show()
	img_y = cv2.resize(cv2.imread('../long/00004_00_10s.jpg',0), (1616,1080))
	plt.imshow(img_y, cmap='gray')
	plt.show()


	start = time.time()
	img_y_hat = cv2.equalizeHist(img_x)
	time_elapsed = time.time() - start
	print('Took {} seconds.'.format(time_elapsed))
	print('Gives a framerate of {}fps.'.format(1/time_elapsed))
	plt.imshow(img_y_hat, cmap='gray')
	plt.show()
