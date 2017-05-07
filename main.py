import scipy.misc as misc
import matplotlib.pyplot as plt
from Filter import *
from Hist import *

if __name__ == '__main__':
    img = misc.imread('Resources/healthy/img_fjords.jpg',0)
    red = img.copy()
    red[:, :, 1] = 0  # green chennel
    red[:, :, 2] = 0  # blue channel
    plt.subplot(1,2,1)
    plt.imshow(red, cmap='gray')
    filtered = gaussianFilter(red)
    transformed = transformImage(filtered)
    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap='gray')
    plt.show()
