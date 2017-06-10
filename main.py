import scipy.misc as misc
import matplotlib.pyplot as plt
from Filter import *
from Hist import *
from teacher import *

if __name__ == '__main__':
    img = misc.imread('Resources/healthy/data',0)
    # red = img.copy()
    make_samples()
    # plt.subplot(1,2,1)
    # plt.imshow(red, cmap='gray')
    # filtered = gaussianFilter(red)
    # transformed = transformImage(filtered)
    # plt.subplot(1, 2, 2)
    # plt.imshow(filtered, cmap='gray')
    # plt.show()


# Zapis wyników
# SK learn / orange