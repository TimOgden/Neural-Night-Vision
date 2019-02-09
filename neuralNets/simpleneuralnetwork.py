import keras
from keras.layers import Dense

class SimpleNeuralNetwork:
	def __init__(self):
		self.model = build_model()

	#Here is where we will define what the model architecture is:
	def build_model():
		model = keras.Sequential([
				Flatten(input_shape=(64,64,1,))
				Dense(4096, activation='relu'), #4096 = 64*64
				Dense(4000, activation='relu'),
				Dense(4000, activation='relu'),
				Dense(4096, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size, epochs, verbose, name):
		# to do:
		# (self.train_x, self.train_y) = get_data()
		# (self.test_x, self.test_y) = get_data()
		# (self.val_x, self.val_y) = get_data()
		self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=[test_x, test_y])
		self.model.save(name + '.h5')

	def test_model(self):
		self.model.evaluate(x=self.val_x, y=self.val_y)

	def predict(self, index=0):
		img = self.model.predict()