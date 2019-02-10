import keras
from keras.layers import Dense, Flatten
from obtain_images import *
import numpy as np
import obtain_images
#from keras.preprocessing.images import ImageDataGenerator
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
class SimpleNeuralNetwork:


	def preprocess(self, x):
		x = x.astype('float32') / 255.
		return x.reshape(-1, np.prod(x.shape[1:])) # flatten

	#Here is where we will define what the model architecture is:
	def build_model(self):
		neuron_count = self.x_res*self.y_res*self.n_channels
		model = keras.Sequential([
				Dense(neuron_count, activation='relu', input_shape=(neuron_count,)),
				Dense(neuron_count, activation='relu'),
				Dense(neuron_count, activation='relu'),
				Dense(neuron_count, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size=3, epochs=10, verbose=1, amount=-1):
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
		self.model.save('SimpleNeuralNetwork-bs{}-ep{}'.format(batch_size,epochs) + '.h5')

	def test_model(self):
		(val_x, val_y) = obtain_data('../val.txt', amount=1, transform=shrink_greyscale_func(self.x_res,self.y_res,self.n_channels))
		val_x = self.preprocess(val_x)
		val_y = self.preprocess(val_y)
		results = self.model.evaluate(x=val_x, y=val_y)
		print('Accuracy was {}'.format(results[1]))

	def predict(self, index=0):
		img = self.model.predict()
		plt.imshow(img)
		plt.show()

	def __init__(self, x_res=1080, y_res=1616, n_channels=3):
		self.x_res = x_res
		self.y_res = y_res
		self.n_channels = n_channels
		self.model = self.build_model()

if __name__ == '__main__':
	neuralNet = SimpleNeuralNetwork(x_res=32, y_res=32, n_channels=1)
	neuralNet.fit_model()
	#neuralNet.test_model()
