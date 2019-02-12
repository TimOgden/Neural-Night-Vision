# Let's try this one last time
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
print('Available gpus:',K.tensorflow_backend._get_available_gpus())
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import math
import tensorflow as tf
import cv2
import numpy as np
from statistics import mean
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
			Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
			Conv2D(self.n_channels, (1, 1), padding='same', activation='sigmoid', kernel_initializer=keras.initializers.he_normal())
		])
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-5), loss='mean_squared_error', metrics=['mean_squared_error'])
		print(model.summary())
		return model

	def fit_model(self):
		num_batches = math.ceil(1863/self.batch_size)
		for epoch in range(self.epochs):
			print('Epoch:', epoch+1)
			batch_loss = []
			for batch in range(num_batches):
				x_train, y_train = self.get_batch()
				print('original shapes:', x_train.shape, y_train.shape)
				x_train.reshape(-1,self.x_res,self.y_res,self.n_channels)
				#print('reshaped x_train:', x_train.shape)
				y_train.reshape(-1,self.x_res,self.y_res,self.n_channels)
				loss = self.model.fit_generator(self.datagen_trainx.flow(x_train, y_train), verbose=1)
				batch_loss.append(loss[0])
			self.model.save('cnn-epoch{}'.format(epoch+1))
			print('Epoch loss:',mean(batch_loss))



	def get_batch(self):
		line_num_start = self.current_batch_index
		line_num_end = self.current_batch_index+self.batch_size
		x_train, y_train = self.get_files(line_num_start,line_num_end)
		img_x_train = []
		img_y_train = []
		for i in range(len(x_train)):
			try:
				img_x = cv2.imread(x_train[i])
				img_y = cv2.imread(y_train[i][:-1])  # y_train will have a \n or ' ' at end, so must remove it
				
				img_x_train.append(img_x)
				img_y_train.append(img_y)
			except IndexError as e:
				pass

		self.current_batch_index+=self.batch_size
		return np.array(img_x_train), np.array(img_y_train)

	def get_files(self, line_num_start, line_num_end):
		with open('../new_train.txt','r') as f:
			lst = f.readlines()
			x_train = []
			y_train = []
			for i in range(line_num_start, line_num_end):
				space = lst[i].index(' ')
				x_train.append(lst[i][:space])
				y_train.append(lst[i][space+1:])
			return x_train, y_train

	def __init__(self, batch_size, epochs, gpus=1):
		self.init_datagens()
		self.batch_size = batch_size
		self.epochs = epochs
		self.current_batch_index = 0
		self.x_res = 1080
		self.y_res = 1616
		self.n_channels = 3
		self.gpus = gpus
		self.model = self.build_model()
		self.fit_model()
if __name__=='__main__':
	cnn = None
	with tf.device('/cpu:0'):
		cnn = Dark_Image_CNN(8,10, gpus=1)

	cnn = multi_gpu_model(cnn,gpus=g)
	cnn.fit_model()