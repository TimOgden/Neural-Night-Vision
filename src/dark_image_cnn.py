# Let's try this one last time
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
print('Available gpus:',K.tensorflow_backend._get_available_gpus())
from keras.layers import Conv2D, Dropout, UpSampling2D, MaxPooling2D
import math
import numpy as np
import tensorflow as tf
import cv2
from statistics import mean
import matplotlib.pyplot as plt
import random
import sys
import time

class Dark_Image_CNN:

	def build_model(self):
		model = keras.Sequential([
			UpSampling2D((2,2), input_shape=(self.x_res,self.y_res,self.n_channels)),
			Conv2D(64, (3,3), padding='same', activation='relu', data_format="channels_last", kernel_initializer=keras.initializers.he_normal()),
			MaxPooling2D((2,2)),
			Conv2D(self.n_channels, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal())
		])
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001), loss='mean_squared_error', metrics=['mean_squared_error'])
		print(model.summary())
		return model

	def fit_model_manual(self):

		num_batches = math.ceil(self.num_training_samples/self.batch_size)
		for epoch in range(self.epochs):
			losses = []
			acc = []
			for batch in range(int(num_batches)):
				start_time = time.time()
				percent = batch/num_batches*100
				print('Batch {} - {:.1f} percent done with epoch {}'.format(batch,percent,epoch+1))
				x_train, y_train = self.get_batch()
				print('original shapes:', x_train.shape, y_train.shape)
				x_train.reshape(-1,self.x_res,self.y_res,self.n_channels)
				#print('reshaped x_train:', x_train.shape)
				y_train.reshape(-1,self.x_res,self.y_res,self.n_channels)
				loss = self.model.train_on_batch(x_train, y_train)
				losses.append(loss[0])
				acc.append(loss[1])
				if batch % 10 == 0:
					plt.plot(losses)
					plt.plot(acc)
					plt.show(block=False)
					plt.pause(7)
					plt.close()
				time_elapsed = time.time() - start_time
				print('Loss:', loss[0], '- time:',time_elapsed,'seconds')
			np.save(np.array(losses), 'loss-epoch{}'.format(epoch))
			np.save(np.array(acc), 'acc-epoch{}'.format(acc))
			self.model.save('cnn-epoch{}'.format(epoch+1+self.last_epoch))

	def fit_model(self):
		self.model.fit_generator(self.generate_arrays_from_file('../new_train.txt'), 
			steps_per_epoch=math.ceil(self.num_training_samples/self.batch_size), epochs=10)

	def get_batch(self):
		img_x_train = []
		img_y_train = []
		x_train, y_train = self.get_files()
		for i in range(len(x_train)):
			try:
				img_x = cv2.cvtColor(cv2.resize(cv2.imread(x_train[i]), (1616,1080)), cv2.COLOR_BGR2GRAY) / 255.
				img_y = cv2.cvtColor(cv2.resize(cv2.imread(y_train[i]), (1616,1080)), cv2.COLOR_BGR2GRAY) / 255.
				#img_x = cv2.resize(cv2.imread(x_train[i]), (1616,1080)) / 255.
				#img_y = cv2.resize(cv2.imread(y_train[i]), (1616,1080)) / 255.
				img_x = np.expand_dims(img_x, axis=3)
				img_y = np.expand_dims(img_y, axis=3)
				img_x_train.append(img_x)
				img_y_train.append(img_y)
			except IndexError as e:
				pass
		return np.array(img_x_train), np.array(img_y_train)

	def get_files(self):
		with open('../new_train.txt','r') as f:
			lst = f.readlines()
			x_train = []
			y_train = []
			for i in range(self.batch_size):
				
				rand_el = None
				while rand_el is None:
					try:
						rand_index = random.randint(0,len(self.unused_files))
						rand_el = self.unused_files[rand_index]
					except IndexError as e:
						print('Got index error, trying again')
				
				self.unused_files.remove(rand_el)	
			return x_train, y_train

	def get_file_len(self, file):
		with open(file,'r') as f:
			lst = f.readlines()
			return len(lst)

	def generate_arrays_from_file(self,path):
		while True:
			with open(path) as f:
				for line in f:
					# create numpy arrays of input data
					# and labels, from each line in the file
					x1, y = self.process_line(line)
					yield ({'up_sampling2d_1_input': x1}, {'conv2d_2': y})

	def process_line(self,line):
		space = line.index(' ')
		x_train = line[:space].strip()
		y_train = line[space+1:].strip()
		img_x = cv2.resize(cv2.imread(x_train), (1616,1080)) / 255.
		img_y = cv2.resize(cv2.imread(y_train), (1616,1080)) / 255.
		img_x = np.reshape(img_x, (-1,self.x_res,self.y_res,self.n_channels))
		img_y = np.reshape(img_y, (-1,self.x_res,self.y_res,self.n_channels))
		return img_x, img_y

	def __init__(self, batch_size, epochs, gpus=1, last_epoch=0):
		self.batch_size = batch_size
		self.unused_files = list(range(int(self.get_file_len('../new_train.txt'))))
		self.epochs = epochs
		self.current_batch_index = 0
		self.x_res = 1080
		self.y_res = 1616
		self.n_channels = 1
		self.num_training_samples = 1863
		self.gpus = gpus
		self.model = self.build_model()
		self.last_epoch = last_epoch
		if self.last_epoch>0:
			self.model = load_model('cnn-epoch{}'.format(self.last_epoch+1))

if __name__=='__main__':
	cnn = None
	last_epoch = None
	batch_size = 64
	print(batch_size)
	try:
		last_epoch = int(sys.argv[0])
	except:
		pass
	with tf.device('/cpu:0'):
		if last_epoch is not None:
			cnn = Dark_Image_CNN(batch_size,10, gpus=1, last_epoch=last_epoch)
		else:
			cnn = Dark_Image_CNN(batch_size,10,gpus=1,last_epoch=0)
		

	#cnn = multi_gpu_model(cnn,gpus=1)
	cnn.fit_model()
	print('done')