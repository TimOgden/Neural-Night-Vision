import cv2
import matplotlib.pyplot as plt
import numpy as np
from paper_cnn import Paper_CNN
import time
from cv2 import VideoCapture

def load_image(model, path):
	img = cv2.imread(path)
	img = cv2.resize(img, (1616,1080))

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

def take_pic(model):
	cam = VideoCapture(0)
	s, img = cam.read()
	if s:
		img = cv2.resize(img, (1616,1080))
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

def stream(model, recordingTime = 10):
	cam = VideoCapture(0)
	# Check if camera opened successfully
	if (cam.isOpened() == False): 
		print("Unable to read camera feed")
	 
	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1616,1080))
	orig = cv2.VideoWriter('orig.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1616,1080))
	start = time.time()

	while(True):
		ret, frame = cam.read()
		if ret == True:
			frame = cv2.resize(frame, (1616, 1080))
			orig.write(frame)
			# Write the frame into the file 'output.avi'
			frame = np.reshape(frame, (1, 1080, 1616, 3))/255.
			yhat = model.predict(frame)
			yhat = np.uint8(255 * yhat)
			yhat = np.reshape(yhat, (1080, 1616, 3))
			out.write(yhat)
		
			if time.time() - start >= recordingTime:
				break
	
		# Break the loop
		else:
			break 
	# When everything done, release the video capture and video write objects
	cam.release()
	out.release()
	 
	# Closes all the frames
	cv2.destroyAllWindows()
if __name__=='__main__':
	model = Paper_CNN(1080,1616,3, 'paper_model')
	model.model = model.build_small_model()
	model.load_model('./weights/small_model_best.h5')
	load_image(model, '../Sony/test.jpg')
	#take_pic(model)
	#stream(model, recordingTime = 30)