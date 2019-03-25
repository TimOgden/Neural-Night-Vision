import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('../myimages/short_001.jpg', 0)
plt.imshow(cv2.equalizeHist(img), cmap='gray')
plt.show()