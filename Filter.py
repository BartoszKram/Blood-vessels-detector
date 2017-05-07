import numpy as np


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
    x_dim = image.shape[1]
    y_dim = image.shape[0]
    for i in range (2,y_dim-2):
        for j in range(2,x_dim-2):
            filtered_image[i][j] = weightSum(image,gauss,i,j)
    return filtered_image
