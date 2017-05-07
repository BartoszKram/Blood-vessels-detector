import scipy.misc as misc
import matplotlib.pyplot as plt
from Filter import *

if __name__ == '__main__':
    img = misc.imread('Resources/healthy/img_fjords.jpg')
    plt.subplot(1,2,1)
    plt.imshow(img)
    filtered = gaussianFilter(img)
    plt.subplot(1,2,2)
    plt.imshow(filtered)
    plt.show()