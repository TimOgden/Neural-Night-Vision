import keras
from keras.layers import Dense


class AutoEncoder:
	def __init__(self, decay_rate):
		self.decay_rate = decay_rate
		if decay_rate<=0 or decay_rate>=1:
			raise ValueError('Decay rate must be above 0 and below 1.')
		self.model = build_model()

	#Here is where we will define what the model architecture is:
	def build_model(self, im_dim):
		flatten_size = im_dim*im_dim
		model = keras.Sequential([
				Flatten(input_shape=(im_dim,im_dim,1,)),
				Dense(flatten_size, activation='relu'), #4096 = 64*64
				Dense(int(flatten_size*decay_rate), activation='relu'),
				Dense(int(flatten_size*(decay_rate**2)), activation='relu'),
				Dense(int(flatten_size*decay_rate), activation='relu'),
				Dense(flatten_size, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size, epochs, verbose, name):
		# to do:
		# (self.train_x, self.train_y) = get_data()
		# (self.test_x, self.test_y) = get_data()
		# (self.val_x, self.val_y) = get_data()
		self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
			steps_per_epoch=len(x_train) / batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_data=[test_x, test_y])
		self.model.save('AutoEncoder-bs{}-ep{}'.format(batch_size,epochs) + '.h5')

	def test_model(self, test_array=None):
		#if test_array:
		#	for element in test_array:
				#
		#else:
		vals = self.model.evaluate(x=self.val_x, y=self.val_y)
		print('Accuracy is {}'.format(vals[1])) # Not sure if this is right yet

	def predict(self, index=0):
		img = self.model.predict()
		plt.imshow(img)
		plt.show()

