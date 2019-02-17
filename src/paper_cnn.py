# Let's try this one last time
import os
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dropout, UpSampling2D, MaxPooling2D, LeakyReLU, Lambda
import math
import numpy as np
import tensorflow as tf
import cv2
from statistics import mean
import matplotlib.pyplot as plt
import random
import sys
import time

class Large_Dark_Image_CNN:

	def build_model(self):
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels)),
				LeakyReLU(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(512, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(512, (3,3), padding='same'),
				LeakyReLU(),

				UpSampling2D(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),

				UpSampling2D(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),

				UpSampling2D(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),

				UpSampling2D(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),

				Conv2D(12, (1,1), padding='same', activation=None),
				Lambda(self.depth_to_space)
			])
		self.callback = ModelCheckpoint('paper_model_weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0, mode='min')
		self.callback = ModelCheckpoint('paper_model_weights.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-7), loss='mean_absolute_error', metrics=['mean_absolute_error'])
		print(model.summary())
		return model


	def fit_model(self, batch_size, epochs, initial_epoch):
		self.model.fit_generator(self.generate_arrays_from_file('../new_train.txt'), 
			steps_per_epoch=math.ceil(self.num_training_samples/batch_size), epochs=epochs, initial_epoch=initial_epoch,
			validation_data=self.generate_arrays_from_file('../val.txt'), validation_steps=26, callbacks=[self.callback])

	def load_model(self, file):
		self.model = load_model(file)

	def generate_arrays_from_file(self,path):
		while True:
			with open(path) as f:
				for line in f:
					# create numpy arrays of input data
					# and labels, from each line in the file
					x1, y = self.process_line(line)
					for x_batch, y_batch in self.datagen.flow(x1,y):
						yield ({'conv2d_1_input': x_batch}, {'lambda_1': y_batch})

	def process_line(self,line):
		space = line.index(' ')
		x_train = line[:space].strip()
		y_train = line[space+1:].strip()
		img_x = cv2.resize(cv2.imread(x_train), (1616,1080)) / 255.
		img_y = cv2.resize(cv2.imread(y_train), (1616,1080)) / 255.
		img_x = np.reshape(img_x, (-1,self.x_res,self.y_res,self.n_channels))
		img_y = np.reshape(img_y, (-1,self.x_res,self.y_res,self.n_channels))
		return img_x, img_y

	def predict(self, img):
		return self.model.predict(img)

	def depth_to_space(self, input_tensor):
		return tf.image.resize_bilinear(tf.depth_to_space(input_tensor, 2), (1080,1616))

	def __init__(self, x_res, y_res, n_channels):
		self.x_res = x_res
		self.y_res = y_res
		self.n_channels = n_channels
		self.num_training_samples = 1863
		self.model = self.build_model()
		self.datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)

if __name__=='__main__':
	cnn = None
	initial_epoch = 0
	batch_size = 64
	num_epochs = 4000
	print(batch_size)

	with tf.device('/cpu:0'):
		cnn = Large_Dark_Image_CNN(1080, 1616, 3)
	cnn.fit_model(batch_size, num_epochs, initial_epoch)
	print('done')