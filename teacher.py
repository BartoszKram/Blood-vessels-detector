import cv2
import numpy as np
import scipy.misc

def make_samples():
    for i in range(1,5):
        image = scipy.misc.imread('Resources/training/images/img'+str(i))
        mask = scipy.misc.imread('Resources/training/masks/img' + str(i)+'_mask')
        x_dim, y_dim, c = image.shape
        for y in range (2,y_dim-3,5):
            for x in range (2,x_dim-3,5):
                sample = image[x-2:x+3,y-2:y+3]
                sample_mask = mask[x-2:x+3,y-2:y+3]
                if(test_sample(sample_mask)):
                    scipy.misc.imsave('Resources/training/positive/sample' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg', sample)
                else:
                    scipy.misc.imsave('Resources/training/false/sample' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg', sample)

def test_sample(sample_mask):
    x_dim, y_dim = sample_mask.shape
    val=0
    for y in range(int(y_dim / 2) - 2, int(y_dim / 2) + 2):
        for x in range(int(x_dim/2)-2, int(x_dim/2)+2):
            val+= sample_mask[x,y]
    if(val>6):
        return True
    else:
        return False