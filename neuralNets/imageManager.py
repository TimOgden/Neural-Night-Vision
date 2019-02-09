import numpy as np

class ImageManager:
	def __init__(self):

	def preprocess(self, img):

	@staticmethod
	def postprocess(self, img, img_shape=(64,64,1)):
		# Unflatten
		img = img.reshape(img_shape)
		return img