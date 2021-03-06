import keras
from keras.layers import Dense, Flatten
from keras.models import load_model
from obtain_images import *
import numpy as np
import obtain_images
import matplotlib.pyplot as plt
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

class SimpleNeuralNetwork:


	def preprocess(self, x):
		x = x.astype('float32') / 255.
		return x.reshape(-1, np.prod(x.shape[1:])) # flatten

	#Here is where we will define what the model architecture is:
	def build_model(self):
		model = keras.Sequential([
				Dense(3600, activation='relu', input_shape=(3600,)), #4096 = 64*64
				Dense(3200, activation='relu'),
				Dense(3200, activation='relu'),
				Dense(3200, activation='relu'),
				Dense(3200, activation='relu'),
				Dense(3200, activation='relu'),
				Dense(3200, activation='relu'),
				Dense(3600, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size=1, epochs=10, verbose=1, amount=-1):
		# to do:
		print('Hello')
		(train_x, train_y) = obtain_data('../train.txt', amount=amount, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))
		(test_x, test_y) = obtain_data('../test.txt', amount=amount, transform=shrink_greyscale_func(self.x_res, self.y_res, self.n_channels))
		train_x = self.preprocess(train_x)
		train_y = self.preprocess(train_y)
		test_x = self.preprocess(test_x)
		test_y = self.preprocess(test_y)
		#self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
			#steps_per_epoch=len(x_train) / batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_data=[test_x, test_y])
		self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=verbose)
		self.model.save('SimpleNeuralNetwork-bs{}-ep{}-({},{})'.format(batch_size,epochs,self.x_res,self.y_res) + '.h5')

	def test_model(self):
		(val_x, val_y) = obtain_data('../val.txt', amount=-1, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))
		val_x = preprocess(val_x)
		val_y = preprocess(val_y)
		results = self.model.evaluate(x=val_x, y=val_y)
		print('Accuracy was {}'.format(results[1]))

	def predict(self, index):
		(test_x, _) = obtain_data('../test.txt', amount=-1, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))
		#print(test_x)
		

		plt.imshow(test_x[index], cmap='gray')
		plt.show()
		#print(test_x.shape)
		test_x = self.preprocess(test_x)
		print(test_x[index].shape)
		val = test_x[index].reshape(-1, test_x[index].shape[0])
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

	def load_model(self, name):
		self.model = load_model(name)
if __name__ == '__main__':
	neuralNet = SimpleNeuralNetwork(x_res=60, y_res=60, n_channels=1)
	neuralNet.fit_model(amount=500)
	#neuralNet.load_model('SimpleNeuralNetwork-bs1-ep10-(60,60).h5')
	neuralNet.predict(index=25)
	neuralNet.predict(index=5)


