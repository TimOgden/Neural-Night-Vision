import keras
from keras.layers import Dense, Conv2D

class SimpleCNN:
	def __init__(self, strides):
		self.strides = strides
		self.model = build_model()		

	#Here is where we will define what the model architecture is:
	def build_model(self, im_dim):
		model = keras.Sequential([
				Conv2D(im_dim, kernel_size=2, strides=self.strides, activation='relu', input_shape=(im_dim,im_dim,1,)),
				Conv2D(im_dim, kernel_size=2, strides=self.strides, activation='relu'),
				Conv2D(im_dim, kernel_size=2, strides=self.strides, activation='relu'),
				Flatten(),
				Dense(im_dim*im_dim, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size, epochs, verbose, name):
		# to do:
		# (self.train_x, self.train_y) = get_data()
		# (self.test_x, self.test_y) = get_data()
		# (self.val_x, self.val_y) = get_data()
		self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=[test_x, test_y])
		self.model.save('SimpleCNN-bs{}-ep{}'.format(batch_size,epochs) + '.h5')

	def test_model(self, test_array=None):
		if test_array:
			for element in test_array:
				#
		else:
			self.model.evaluate(x=self.val_x, y=self.val_y)
