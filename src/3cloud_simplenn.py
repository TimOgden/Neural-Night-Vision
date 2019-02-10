import keras
from keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D, UpSampling2D
from keras.models import load_model
from obtain_images import *
import numpy as np
import obtain_images
import matplotlib.pyplot as plt
from keras import backend as K
import math
#from statistics import mean
import time
from os.path import exists

print(K.tensorflow_backend._get_available_gpus())

class ConvolutionalNeuralNetwork:


	def preprocess(self, x):
		if self.n_channels == 1:
			x = np.array(x)
			x = np.expand_dims(x, axis=3)
		return x.astype('float32') / 255 # no flatten

	#Here is where we will define what the model architecture is:
	def build_model(self):
		size = self.x_res * self.y_res * self.n_channels
		model = keras.Sequential([
				Conv2D(32, (3,3), padding='same', activation='relu', data_format="channels_last", input_shape=(self.x_res,self.y_res,self.n_channels), kernel_initializer=keras.initializers.he_normal()),
				MaxPooling2D((2,2)),
				Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
				MaxPooling2D((2, 2)),
				Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
                                MaxPooling2D((2, 2)),
                                Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
                                UpSampling2D((2, 2)),
                                Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
				UpSampling2D((2, 2)),	
				Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=keras.initializers.he_normal()),
				UpSampling2D((2, 2)),
				Conv2D(self.n_channels, (1, 1), padding='same', activation='sigmoid', kernel_initializer=keras.initializers.he_normal())
			])
		model.compile(optimizer=keras.optimizers.Adam(lr=.01), loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size=1, epochs=10, verbose=1, amount=-1, track_losses=False):
                        num_batches = math.ceil(1863/batch_size)
                        losses = []
                        prev_loss = 0
                        for epoch in range(self.init_epoch, epochs):
                                recent_losses = []
                                for batch_index in range(int(num_batches)):
                                        (train_x_batch, train_y_batch) = get_batch('../new_train.txt', batch_index, batch_size, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))
                                        train_x_batch = self.preprocess(train_x_batch)
                                        #print(train_x_batch.shape)
                                        train_y_batch = self.preprocess(train_y_batch)
                                        start = time.time()
                                        loss = self.model.train_on_batch(x=train_x_batch, y=train_y_batch)
                                        print('time {}'.format(time.time() - start))
                                        train_y_batch = None
                                        train_x_batch = None
                                        print('Epoch {} of {}, batch {} of {}'.format(epoch+1,epochs,batch_index+1,num_batches))
                                        print('Loss was {}'.format(loss[0]))
                                        print('Change in loss: {}'.format(loss[0] - loss))
                                        prev_loss = loss[0]
                                        print('Accuracy was {}'.format(loss[1]))
                                        recent_losses.append(loss[0])
                                        losses.append(loss[0])
                                if epoch:
                                    self.model.save_weights('ConvolutionalNeuralNetwork-bs{}-ep{}-{}'.format(batch_size,epoch,epochs) + '.h5')
                                if epoch % 20 == 0:
                                    if track_losses:
                                        pass

	def test_model(self):
		(val_x, val_y) = obtain_data('../val.txt', amount=-1, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))
		val_x = preprocess(val_x)
		val_y = preprocess(val_y)
		results = self.model.evaluate(x=val_x, y=val_y)
		print('Accuracy was {}'.format(results[1]))

	def predict(self, index):
		(test_x, _) = obtain_data('../new_test.txt', amount=-1, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))

		plt.imshow(test_x[index], cmap='gray')
		plt.show()
		
		test_x = self.preprocess(test_x)
		print(test_x[index].shape)
		val = test_x[index]
		val = np.expand_dims(val, axis=0)
		print(val.shape)
		img = self.model.predict(val)
		img = img.reshape(self.x_res, self.y_res)
		plt.imshow(img, cmap='gray')
		plt.show()

	def __init__(self, x_res=1080, y_res=1616, n_channels=3):
		self.x_res = x_res
		self.y_res = y_res
		self.n_channels = n_channels
		self.model = self.build_model()
		self.val_x = None
		self.val_y = None
		self.init_epoch = 0

	def load_model(self, name, epoch):
		self.model.load_weights(name)
		self.init_epoch = epoch

if __name__ == '__main__':
    neuralNet = ConvolutionalNeuralNetwork(x_res=128, y_res=128, n_channels=1)
    neuralNet.load_model('ConvolutionalNeuralNetwork-bs2000-ep6-10.h5', 7)
    neuralNet.fit_model(epochs=10, batch_size=2000)
