import keras
from keras.layers import Dense, Conv2D

class SimpleCNN:
	def __init__(self):
		self.model = build_model()

	
	#Here is where we will define what the model architecture is:
	def build_model():
		model = keras.Sequential([
				Conv2D(64, kernel_size=2, activation='relu', input_shape=(64,64,1,)),
				Conv2D(64, kernel_size=2, activation='relu'),
				Conv2D(64, kernel_size=2, activation='relu'),
				Flatten(),
				Dense(4096, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	model = build_model()



	def fit_model(batch_size, epochs, verbose, name):
		# to do:
		# (train_x, train_y) = get_data()
		# (test_x, test_y) = get_data()
		model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=[test_x, test_y])
		model.save(name + '.h5')

	def test_model()
