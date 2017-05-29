import scipy.misc as misc
import matplotlib.pyplot as plt
from Filter import *
from Hist import *
import cv2
from colorsys import hsv_to_rgb

if __name__ == '__main__':


    #img = misc.imread('resources/01_h.jpg',0)
    #red = img.copy()
    #red[:, :, 1] = 0  # green chennel
    #red[:, :, 2] = 0  # blue channel
    #plt.subplot(1,2,1)
    #plt.imshow(red, cmap='gray')
    #filtered = gaussianFilter(red)
    #transformed = transformImage(filtered)
    #plt.subplot(1, 2, 2)
    #plt.imshow(transformed, cmap='gray')
    #plt.show()


    names = ["resources/%02d_h.jpg" % x for x in range(1,16)]
    for i, name in enumerate(names):
      img = cv2.imread(name)
      kernel = np.ones((2,2)*3)
      temp = cv2.multiply(img, np.array([1.4]))
      temp = cv2.GaussianBlur(temp, (5,5), 0)
      temp = cv2.medianBlur(temp,5)
      #temp = cv2.bilateralFilter(temp,9,75,75)
      temp = cv2.Canny(temp,25,25)
      temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
      temp = cv2.dilate(temp, np.ones((3,3)*2), iterations=5)
      #temp = cv2.erode(temp,kernel,iterations = 1)
      temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

      _, k, tmp = cv2.findContours(image = temp, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
      for j,kontur in enumerate(k):
          cv2.drawContours(img, [kontur],0,np.array(hsv_to_rgb(1.0, 1.0, 1.0))*255.0,2)
          #moments = cv2.moments(kontur)

      #median = cv2.medianBlur(img,5)
      cv2.imwrite("results/%02d_h_detect.jpg" % i, img)


