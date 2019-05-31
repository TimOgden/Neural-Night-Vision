# Let's try this one last time
import os
import keras
from keras.models import load_model, Model
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import *
from keras.optimizers import Adam
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

	def build_unet(self, pretrained_weights=None, input_size=(1080, 1616, 3), dropout=.5):
		inputs = Input(input_size, name='input_input')
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1a')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1b')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2a')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2b')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3a')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3b')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4a')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4b')(conv4)
		drop4 = Dropout(dropout, name='drop4')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5a')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5b')(conv5)
		drop5 = Dropout(dropout, name='drop5')(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv6')(UpSampling2D(size = (2,2), name='up1')(drop5))
		crop = ZeroPadding2D(padding=((1,0),(0,0)))(up6)
		merge6 = concatenate([drop4,crop], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7a')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7b')(conv6)
		drop6 = Dropout(dropout, name='drop6')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		drop7 = Dropout(dropout, name='drop7')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		drop8 = Dropout(dropout, name='drop8')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(3, 1, activation = 'sigmoid', name='output')(conv9)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr=1e-4, decay=1e-5), loss = 'mean_absolute_error')
		
		#model.summary()

		if(pretrained_weights):
			model.load_weights(pretrained_weights)

		return model

	def build_model(self):
		dropout = 0.5
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels), name='input'),
				LeakyReLU(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(512, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(512, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(256, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(128, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				BatchNormalization(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				Conv2D(32, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				Conv2D(12, (1,1), padding='same', activation=None),
				Lambda(self.depth_to_space, name='output')
				#Reshape((self.x_res,self.y_res,self.n_channels))
			])
		
		model.compile(optimizer=Adam(lr=1e-4, decay=1e-6), loss='mean_absolute_error')
		#print(model.summary())
		return model

	def build_med_model(self):
		dropout = 0
		model = keras.Sequential([
				Conv2D(64, (3,3), padding='same', input_shape=(self.x_res, self.y_res, self.n_channels), name='input'),
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

				Conv2D(12, (1,1), padding='same', activation=None),
				Lambda(self.depth_to_space, name='output')
			])
		
		model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss='mean_absolute_error')
		#print(model.summary())
		return model

	def build_small_model(self):
		dropout = .25
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels), name='input'),
				LeakyReLU(),
				Dropout(dropout),
				MaxPooling2D((2,2), padding='same'),

				Conv2D(64, (3,3), padding='same'),
				LeakyReLU(),
				Dropout(dropout),

				UpSampling2D(),
				Conv2D(3, (3,3), padding='same', name='output'),
			])
		model.compile(optimizer=keras.optimizers.SGD(lr=.0001, nesterov=True, decay=1e-5), loss='mean_absolute_error')
		#print(model.summary())
		return model

	def build_smallest(self):
		dropout = .5
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', input_shape=(self.x_res,self.y_res,self.n_channels)),
				Conv2D(3, (3,3), padding='same'),
			])
		model.compile(optimizer=keras.optimizers.Adam(lr=.0001), loss='mean_absolute_error')
		#print(model.summary())
		return model

	def show_output(self, im_1, im_2):
		print(im_1.shape)
		for i in range(self.batch_size-1):
			#im_1 = np.reshape(im_1, (im_1.shape[1], im_1.shape[2], im_1.shape[3]))
			plt.subplot(1,2,1)
			plt.imshow(im_1[i]/255.)
			

			#im_2 = np.reshape(im_2, (im_2.shape[1], im_2.shape[2], im_2.shape[3]))
			plt.subplot(1,2,2)
			plt.imshow(im_2[i]/255.)
			plt.show()

	def show_output_single(self, im_1):
		print(im_1.shape)
		for i in range(self.batch_size-1):
			plt.imshow(im_1[i]/255.)
			plt.show()

	def fit_model(self, batch_size, epochs, initial_epoch, callbacks):
		seed = 1337
		short_generator = self.datagen.flow_from_directory('../screenshots/short/', class_mode=None,
			target_size=(self.x_res,self.y_res), subset='training', batch_size=self.batch_size, seed=seed, shuffle=True)
		long_generator = self.datagen.flow_from_directory('../screenshots/long/', class_mode=None,
			target_size=(self.x_res,self.y_res), subset='training', batch_size=self.batch_size, seed=seed, shuffle=True)
	
		short_val = self.datagen.flow_from_directory('../screenshots/short/', class_mode=None,
			target_size=(self.x_res,self.y_res), subset='validation', batch_size=self.batch_size, seed=seed, shuffle=True)
		long_val = self.datagen.flow_from_directory('../screenshots/long/', class_mode=None,
			target_size=(self.x_res,self.y_res), subset='validation', batch_size=self.batch_size, seed=seed, shuffle=True)
		print('zipping generators')
		generator = zip(short_generator, long_generator)
		val_gen = zip(short_val, long_val)
		print('done zipping generators')
		#self.show_output(*next(generator))
		#self.show_output(next(short_generator), next(long_generator))
		self.model.fit_generator(generator, steps_per_epoch=math.ceil(5114/self.batch_size), epochs=epochs, 
			validation_data=val_gen, validation_steps=math.ceil(1279/self.batch_size), callbacks=self.callbacks)
		self.model.save('./weights/finished.h5')
		#self.model.fit_generator(self.generate_arrays_from_file('../unity_train.txt'), 
		#	steps_per_epoch=math.ceil(4399/(self.batch_size)), epochs=epochs,
		#	validation_data=self.generate_arrays_from_file('../unity_test.txt', datagen=None), 
		#	validation_steps=math.ceil(1100/self.batch_size), callbacks=self.callbacks)
	


	def load_model(self, file):
		self.model = load_model(file)

	def generate_arrays_from_file(self,path,datagen=None):
		while True:

			with open(path) as f:
				c = 0
				x_vals = []
				y_vals = []
				lines = f.readlines()

				for i in range(c, c+self.batch_size):

					length = len(lines)
					if i >= length:
						i -= length # wrap around

					x1, y = self.process_line(lines[i])
					x_vals.append(x1)
					y_vals.append(y)

			if datagen is not None:
				(x_batch, y_batch) = next(datagen.flow_from_directory('screenshots', batch_size=self.batch_size, shuffle=True))
				#print('Hello!')
				yield ({'input_input': x_batch}, {'output': y_batch})
			else:
				yield ({'input_input': np.array(x_vals)}, {'output': np.array(y_vals)})


	def process_line(self,line):
		space = line.index(' ')
		x_train = line[:space].strip()
		y_train = line[space+1:].strip()
		img_x = cv2.imread(x_train)
		img_y = cv2.imread(y_train)
		if img_x is None or img_y is None:
			print('img x is none:', x_train, '\nimg y is none:', y_train)
			return None, None
		img_x = cv2.resize(img_x, (self.x_res,self.y_res)) / 255.
		img_y = cv2.resize(img_y, (self.x_res,self.y_res)) / 255.
		img_x = np.reshape(img_x, (self.x_res,self.y_res,self.n_channels))
		img_y = np.reshape(img_y, (self.x_res,self.y_res,self.n_channels))
		return img_x, img_y


	def predict(self, img):
		return self.model.predict(img)

	def depth_to_space(self, input_tensor):
		return tf.image.resize_bilinear(tf.depth_to_space(input_tensor, 2), (self.x_res,self.y_res))

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

	def file_len(self, fname):
		return sum(1 for line in open(fname))
	
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

	def __init__(self, x_res, y_res, n_channels, name, batch_size):
		self.x_res = x_res
		self.y_res = y_res
		self.n_channels = n_channels
		self.batch_size = batch_size
		#self.model = self.build_model()
		#print('Model memory usage:', self.get_model_memory_usage(1,self.model))
		#self.datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10, width_shift_range=.2, height_shift_range=.2, rescale=1/255., validation_split=.2)
		self.datagen = ImageDataGenerator(validation_split=.2, horizontal_flip=True, rotation_range=10, width_shift_range=.2, height_shift_range=.2)
		self.save_best = ModelCheckpoint('./weights/'+ name + '_best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min')
		self.checkpoint = ModelCheckpoint('./weights/'+ name + '_chkpt_{epoch:04d}.h5', monitor='val_loss', save_best_only=False, verbose=1, mode='min', period=5)
		self.tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=self.batch_size)
		self.lr_schedule = LearningRateScheduler(self.lr_sched)
		self.callbacks = [self.checkpoint, self.tensorboard, self.save_best]

if __name__=='__main__':
	

	initial_epoch = 0
	batch_size = 16
	num_epochs = 100

	cnn = None
	cnn = Paper_CNN(int(1080/2), int(1616/2), 3, 'working_model', batch_size)
	#cnn = Paper_CNN(128,128, 3, 'working_model', batch_size)
	cnn.model = cnn.build_model()
	print(cnn.model.summary())
	if initial_epoch is not 0:
		cnn.load_model('./weights/paper_model_chkpt_04.h5')

	cnn.fit_model(batch_size, num_epochs, initial_epoch, [cnn.tensorboard, cnn.save_best, cnn.checkpoint])
	print('done')