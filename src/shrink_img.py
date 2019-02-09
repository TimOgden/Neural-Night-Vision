import cv2
import numpy as np

def shrink(img, xres, yres):
    return cv2.resize(img, dsize=(xres, yres), interpolation=cv2.INTER_CUBIC)

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

