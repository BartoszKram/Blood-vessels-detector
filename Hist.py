import matplotlib.pyplot as plt
from math import floor
import cv2
import numpy as np

def calculateDist(hist,size):
    D = np.zeros_like(hist)
    i = 0
    for val in hist:
        D[i] = D[i-1]+val[0]/size
        i += 1
    return D

def calculateLUT(D, d0):
    LUT = np.zeros_like(D)
    for i in range(0, D.shape[0]):
        LUT[i] = floor((D[i]-d0)/(1-d0)*255)
    return LUT

def normalizeHist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256], False)
    x,y,c = image.shape
    D = calculateDist(hist,x*y)
    d0 =0
    for i in range(0, D.shape[0]):
        if(D[i][0]!=0):
            d0=D[i][0]
            break
    LUT = calculateLUT(D, d0)
    return LUT

def transformImage(image):
    LUT = normalizeHist(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            image[i][j][0] = LUT[image[i][j]][0]
    return image
