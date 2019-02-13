# Let's try this one last time
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
print('Available gpus:',K.tensorflow_backend._get_available_gpus())
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import math
import tensorflow as tf
import cv2
import numpy as np
from statistics import mean
import random
import sys

class Dark_Image_CNN:
	def init_datagens(self):
		self.datagen_trainx = ImageDataGenerator(
			rescale=1./255,
			rotation_range=10,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)
		self.datagen_trainy = ImageDataGenerator(
			rescale=1./255,
			rotation_range=10,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)

	def build_model(self):
		model = keras.Sequential([
			Conv2D(64, (3,3), padding='same', activation='relu', data_format="channels_last", input_shape=(self.x_res,self.y_res,self.n_channels), kernel_initializer=keras.initializers.he_normal()),
			MaxPooling2D((2,2)),
			Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
			MaxPooling2D((2, 2)),
			Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
			UpSampling2D((2, 2)),
			Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
			UpSampling2D((2, 2)),
			Conv2D(self.n_channels, (1, 1), padding='same', activation='sigmoid', kernel_initializer=keras.initializers.he_normal())
		])
		model.compile(optimizer=keras.optimizers.Adam(lr=.00002, decay=1e-5), loss='mean_squared_error', metrics=['mean_squared_error'])
		print(model.summary())
		return model

	def fit_model(self):
		num_batches = math.ceil(1863/self.batch_size)
		for epoch in range(self.epochs):
			for batch in range(int(num_batches)):
				percent = batch/num_batches*100
				print('{:.1f} percent done with epoch {}'.format(percent,epoch+1))
				x_train, y_train = self.get_batch()
				#print('original shapes:', x_train.shape, y_train.shape)
				x_train.reshape(-1,self.x_res,self.y_res,self.n_channels)
				#print('reshaped x_train:', x_train.shape)
				y_train.reshape(-1,self.x_res,self.y_res,self.n_channels)
				self.model.fit_generator(self.datagen_trainx.flow(x_train, y_train), verbose=2, steps_per_epoch=len(x_train)/self.batch_size)

			self.model.save('cnn-epoch{}'.format(epoch+1+self.last_epoch))



	def get_batch(self):
		img_x_train = []
		img_y_train = []
		x_train, y_train = self.get_files()
		for i in range(len(x_train)):
			try:
				img_x = cv2.cvtColor(cv2.resize(cv2.imread(x_train[i]), (1616,1080)), cv2.COLOR_BGR2GRAY)
				img_y = cv2.cvtColor(cv2.resize(cv2.imread(y_train[i]), (1616,1080)), cv2.COLOR_BGR2GRAY)
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
				space = lst[rand_el].index(' ')
				x_train.append(lst[rand_el][:space].strip())
				y_train.append(lst[rand_el][space+1:].strip())
				self.unused_files.remove(rand_el)	
			return x_train, y_train

	def get_file_len(self, file):
		with open(file,'r') as f:
			lst = f.readlines()
			return len(lst)

	def __init__(self, batch_size, epochs, gpus=1, last_epoch=0):
		self.batch_size = batch_size
		self.unused_files = list(range(int(self.get_file_len('../new_train.txt'))))
		self.epochs = epochs
		self.current_batch_index = 0
		self.x_res = 1080
		self.y_res = 1616
		self.n_channels = 1
		self.gpus = gpus
		self.init_datagens()
		self.model = self.build_model()
		self.last_epoch = last_epoch
		if self.last_epoch>0:
			self.model = load_model('cnn-epoch{}'.format(self.last_epoch+1))

if __name__=='__main__':
	cnn = None
	last_epoch = None
	try:
		last_epoch = int(sys.argv[0])
	except:
		pass
	with tf.device('/cpu:0'):
		if last_epoch is not None:
			cnn = Dark_Image_CNN(16,10, gpus=1, last_epoch=last_epoch)
		else:
			cnn = Dark_Image_CNN(16,10,gpus=1,last_epoch=0)

	#cnn = multi_gpu_model(cnn,gpus=1)
	cnn.fit_model()