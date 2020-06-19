import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')

gauss_1 = cv2.GaussianBlur(img,(11,11),2) 
gauss_2 = cv2.GaussianBlur(img,(11,11),5) 
gauss_3 = cv2.GaussianBlur(img,(11,11),10) 

plt.subplot(221),plt.imshow(img),plt.title('original')
plt.subplot(222),plt.imshow(gauss_1),plt.title('sigma : 2')
plt.subplot(223),plt.imshow(gauss_2),plt.title('sigma : 5')
plt.subplot(224),plt.imshow(gauss_3),plt.title('sigma : 10')
plt.savefig('output.jpg')

print('Output imaged saves')

plt.show()
plt.figure()


