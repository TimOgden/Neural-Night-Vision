# Let's try this one last time
import os
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
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

class Large_Dark_Image_CNN:

	def build_model(self):
		model = keras.Sequential([
				Conv2D(32, padding='same', strides=(3,3), activation='leaky_relu', input_shape=(self.x_res,self.y_res,self.n_channels)),
				Conv2D(32, padding='same',strides=(3,3), activation='leaky_relu'),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(64, padding='same',strides=(3,3), activation='leaky_relu'),
				Conv2D(64, padding='same',strides=(3,3), activation='leaky_relu'),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(128, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(128, padding='same', strides=(3,3), activation='leaky_relu'),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(256, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(256, padding='same', strides=(3,3), activation='leaky_relu'),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(512, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(512, padding='same', strides=(3,3), activation='leaky_relu'),

				UpSampling2D(),
				Conv2D(256, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(256, padding='same', strides=(3,3), activation='leaky_relu'),

				UpSampling2D(),
				Conv2D(128, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(128, padding='same', strides=(3,3), activation='leaky_relu'),

				UpSampling2D(),
				Conv2D(64, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(64, padding='same', strides=(3,3), activation='leaky_relu'),

				UpSampling2D(),
				Conv2D(32, padding='same', strides=(3,3), activation='leaky_relu'),
				Conv2D(32, padding='same', strides=(3,3), activation='leaky_relu'),

				Conv2D(12, padding='same', strides=(1,1), activation=None)
			])
		self.callback = ModelCheckpoint('best_larger_model_weights.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
		model.compile(optimizer=keras.optimizers.Adam(lr=.0005, decay=1e-5), loss='mean_absolute_error', metrics=['mean_absolute_error'])
		#print(model.summary())
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

	def fit_model(self, batch_size, epochs):
		self.model.fit_generator(self.generate_arrays_from_file('../new_train.txt'), 
			steps_per_epoch=math.ceil(self.num_training_samples/batch_size), epochs=epochs,
			validation_data=self.generate_arrays_from_file('../new_test.txt'), validation_steps=26, callbacks=[self.callback])

	def generate_arrays_from_file(self,path):
		while True:
			with open(path) as f:
				for line in f:
					# create numpy arrays of input data
					# and labels, from each line in the file
					x1, y = self.process_line(line)
					yield ({'conv2d_1_input': x1}, {'conv2d_7': y})

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

	def __init__(self, x_res, y_res, n_channels):
		self.unused_files = list(range(int(self.get_file_len('../new_train.txt'))))
		self.x_res = x_res
		self.y_res = y_res
		self.n_channels = n_channels
		self.num_training_samples = 1863
		self.model = self.build_model()

if __name__=='__main__':
	cnn = None
	last_epoch = None
	batch_size = 128
	num_epochs = 
	print(batch_size)
	with tf.device('/cpu:0'):
		cnn = Large_Dark_Image_CNN(1080, 1616, 3)
	cnn.fit_model(batch_size, num_epochs)
	print('done')