import keras
import real_simplecnn as nn
from obtain_images import *
import matplotlib.pyplot as plt
if __name__=='__main__':
	neuralNet = nn.ConvolutionalNeuralNetwork(x_res=128,y_res=128,n_channels=1)
	neuralNet.load_model('weights.h5', epoch=20)
	neuralNet.predict(index=0)
	neuralNet.predict(index=25)
