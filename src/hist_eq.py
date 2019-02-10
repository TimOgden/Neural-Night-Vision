from obtain_images import *
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    images = obtain_data('../new_train.txt', amount=1, transform=shrink_greyscale_func(128, 128, 1))
    plt.imshow(images[0][0])
    plt.show() 
    images[0][0] = np.array(cv2.equalizeHist(images[0][0]))
    plt.imshow(images[0][0])
    images[0][0] = images[0][0].astype('float32') / 255
    print(images[0][0].shape)
    plt.show()
