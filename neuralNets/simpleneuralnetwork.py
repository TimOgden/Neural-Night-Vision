import keras
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
class SimpleNeuralNetwork:
	def __init__(self):
		self.model = build_model()

	#Here is where we will define what the model architecture is:
	def build_model():
		model = keras.Sequential([
				Flatten(input_shape=(64,64,1,)),
				Dense(4096, activation='relu'), #4096 = 64*64
				Dense(4000, activation='relu'),
				Dense(4000, activation='relu'),
				Dense(4096, activation='relu')
			])
		model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	def fit_model(self, batch_size=32, epochs=10, verbose=1):
		# to do:
		# (self.train_x, self.train_y) = get_data('Sony_train_list.txt')
		# (self.test_x, self.test_y) = get_data()
		# (self.val_x, self.val_y) = get_data()
		datagen = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)

		self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
			steps_per_epoch=len(x_train) / batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_data=[test_x, test_y])
		#self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=[test_x, test_y])
		self.model.save('SimpleNeuralNetwork-bs{}-ep{}'.format(batch_size,epochs) + '.h5')

	def 
	def test_model(self):
		self.model.evaluate(x=self.val_x, y=self.val_y)

	def predict(self, index=0):
		img = self.model.predict()
		plt.imshow(img)
		plt.show()
