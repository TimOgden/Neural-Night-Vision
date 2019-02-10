import numpy as np
from obtain_images import *
import matplotlib.pyplot as plt

def gamma_conv(img, gamma):
    gamma_inv = 1.0 / gamma
    img = img.astype('float32') / 255
    return np.power(img, gamma_inv) * 255

if __name__ == '__main__':
    images = obtain_data('../new_train.txt', amount=1, transform=shrink_greyscale_func(256, 256, 1))
    plt.show(images[0][0])
    plt.show()
    print(images[0][0].shape)
    images[0][0] = gamma_conv(images[0][0], 2)
    plt.imshow(images[0][0])
    plt.show()


