import cv2
import numpy as np 
import matplotlib.pyplot as plt


img1 = cv2.imread("img.jpg")

scale_percent = 60 
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height) 
img2 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

sift = cv2.xfeatures2d.SIFT_create()
kp_img1, des_img1 = sift.detectAndCompute(img1,None)
kp_img2, des_img2 = sift.detectAndCompute(img2,None)

dst1 = cv2.drawKeypoints(img1,kp_img1,None)
dst2 = cv2.drawKeypoints(img2,kp_img2,None)

plt.imshow(dst1),plt.title('original image')
plt.show()
plt.figure()
cv2.imwrite('output_original.jpg',dst1)
plt.imshow(dst2),plt.title('changed res image')
plt.show()
cv2.imwrite('output_changed_res.jpg',dst2)

print('Output images saved')