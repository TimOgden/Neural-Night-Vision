# Let's try this one last time
import os
import keras
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Conv2D, Dropout, UpSampling2D, MaxPooling2D, LeakyReLU, Lambda, Reshape
import math
import numpy as np
import tensorflow as tf
import cv2
from statistics import mean
import matplotlib.pyplot as plt
import random
import sys
import time
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

class Paper_CNN:

	def build_model(self):
		dropout = .4
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels)),
				LeakyReLU(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(512, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(512, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				Conv2D(12, (1,1), padding='same', activation=None),
				Lambda(self.depth_to_space)
				#Reshape((self.x_res,self.y_res,self.n_channels))
			])
		
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001, decay=5e-6), loss='mean_absolute_error')
		print(model.summary())
		return model

	def build_med_model(self):
		dropout = .4
		model = keras.Sequential([
				Conv2D(64, (3,3), padding='same', input_shape=(self.x_res, self.y_res, self.n_channels)),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				UpSampling2D(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				Conv2D(1, (1,1), padding='same', activation=None),
				Lambda(self.depth_to_space)
			])
		
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001), loss='mean_absolute_error')
		print(model.summary())
		return model

	def build_small_model(self):
		dropout = .3
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels)),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(3, (3,3), padding='same'),
			])
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-5), loss='mean_absolute_error')
		print(model.summary())
		return model

	def build_smallest(self):
		dropout = .5
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels)),
				Conv2D(3, (3,3), padding='same'),
			])
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001), loss='mean_absolute_error')
		print(model.summary())
		return model

	def fit_model(self, batch_size, epochs, initial_epoch, callbacks):
		self.model.fit_generator(self.generate_arrays_from_file('../new_train.txt'), 
			steps_per_epoch=math.ceil(self.num_training_samples/(batch_size)), epochs=epochs, initial_epoch=initial_epoch,
			validation_data=self.generate_arrays_from_file('../val.txt'), validation_steps=26,
			callbacks=callbacks)

	def load_model(self, file):
		self.model = load_model(file)

	def generate_arrays_from_file(self,path):
		while True:
			with open(path) as f:
				xs = np.array([])
				ys = np.array([])
				c = 0
				for line in f:
					c+=1

					# create numpy arrays of input data
					# and labels, from each line in the file
					x1, y = self.process_line(line)
					np.append(xs,x1)
					np.append(ys,y)
					if c % self.batch_size == 0:
						for x_batch, y_batch in self.train_datagen.flow(xs,ys, shuffle=True):
							yield ({'conv2d_1_input': x_batch}, {'conv2d_3': y_batch})
						xs = np.array([])
						ys = np.array([])

	def generate_val_from_file(self, path):
		# Validation data generator
		while True:
			with open(path) as f:
				for line in f:
					# create numpy arrays of input data
					# and labels, from each line in the file
					x1, y = self.process_line(line)
					if x1 is None or y is None:
						continue
					for x_batch, y_batch in self.val_datagen.flow(x1,y, shuffle=True):
						yield ({'conv2d_1_input': x_batch}, {'conv2d_3': y_batch})

	def process_line(self,line):
		space = line.index(' ')
		x_train = line[:space].strip()
		y_train = line[space+1:].strip()
		img_x = cv2.imread(x_train)
		img_y = cv2.imread(y_train)
		if img_x is None or img_y is None:
			print('img x is none:', img_x is None, '\nimg y is none:', img_y is None)
		img_x = cv2.resize(img_x, (1616,1080)) / 255.
		img_y = cv2.resize(img_y, (1616,1080)) / 255.
		img_x = np.reshape(img_x, (-1,self.x_res,self.y_res,self.n_channels))
		img_y = np.reshape(img_y, (-1,self.x_res,self.y_res,self.n_channels))
		return img_x, img_y

	def predict(self, img):
		return self.model.predict(img)

	def depth_to_space(self, input_tensor):
		return tf.image.resize_bilinear(tf.depth_to_space(input_tensor, 2), (1080,1616))

	def reshape(self, input_tensor):
		input_tensor = input_tensor[:1745281]
		return np.reshape(input_tensor, (1080,1616,1))

	def lr_sched(self, epoch):
		top = .00005
		bottom = .000001
		if epoch<2000:
			#return top - epoch*((top-bottom)/(2000-epoch)) # Linear interpolation
			return top
		else:
			return bottom

	def load_model(self, file):
		self.model.load_weights(file)

	def get_model_memory_usage(self,batch_size, model):
		shapes_mem_count = 0
		for l in model.layers:
			single_layer_mem = 1
			for s in l.output_shape:
				if s is None:
					continue
				single_layer_mem *= s
			shapes_mem_count += single_layer_mem

		trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
		non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

		number_size = 4.0
		if K.floatx() == 'float16':
			number_size = 2.0
		if K.floatx() == 'float64':
			number_size = 8.0

		total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
		gbytes = np.round(total_memory / (1024.0 ** 3), 3)
		return gbytes

	def __init__(self, x_res, y_res, n_channels, name):
		self.x_res = x_res
		self.y_res = y_res
		self.n_channels = n_channels
		self.num_training_samples = 1863
		#self.model = self.build_model()
		#print('Model memory usage:', self.get_model_memory_usage(1,self.model))
		self.train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, width_shift_range=.2, height_shift_range=.2)
		self.val_datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=.1, height_shift_range=.1)
		self.save_best = ModelCheckpoint('./weights/'+ name + '_best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min')
		self.checkpoint = ModelCheckpoint('./weights/'+ name + '_chkpt_{epoch:04d}.h5', monitor='val_loss', save_best_only=False, verbose=1, mode='min', period=10)
		self.tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=64)
		self.lr_schedule = LearningRateScheduler(self.lr_sched)

if __name__=='__main__':
	cnn = Paper_CNN(1080, 1616, 3, 'small_layers')

	initial_epoch = 0
	batch_size = 64
	num_epochs = 4000
	print(batch_size)

	cnn.model = cnn.build_small_model()

	if initial_epoch is not 0:
		cnn.load_model('./weights/paper_model_chkpt_04.h5')

	cnn.fit_model(batch_size, num_epochs, initial_epoch, [cnn.tensorboard, cnn.save_best, cnn.checkpoint])
	print('done')