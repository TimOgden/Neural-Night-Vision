import keras
from keras.layers import Dense


#Here is where we will define what the model architecture is:
def build_model():
	model = keras.Sequential([
			Flatten(input_shape=(64,64,3,))
			Dense(4096, activation='relu'), #4096 = 64*64
			Dense(4000, activation='relu'),
			Dense(4000, activation='relu'),
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
