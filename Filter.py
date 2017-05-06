import numpy as np
from scipy.misc import imread

def weightSum(image,gauss,i,j):
    sum = 0.0
    for ii in range (-2,2):
        for jj in range(-2,2):
            sum+=image[i+ii][j+jj]*gauss[ii+2][jj+2]
    weight_sum = sum/52
    return weight_sum

def gaussianFilter(image):
    gauss =   np.array([[1, 1, 2, 1, 1],
                        [1, 2, 4, 2, 1],
                        [2, 4, 8, 4, 2],
                        [1, 2, 4, 2, 1],
                        [1, 1, 2, 1, 1]])
    filtered_image = np.zeros_like(image)
    for i in range (2,filtered_image.size-2):
        for j in range(2,filtered_image.size-2):
            filtered_image[i][j] = weightSum(image,gauss,i,j)
    return filtered_image

if __name__ == '__main__':
    img = imread('')
    gaussianFilter()
