import cv2
import numpy as np
import matplotlib.pyplot as plt

#loading right and left images
r_raw = cv2.imread('right.jpg')
r_img = cv2.cvtColor(r_raw,cv2.COLOR_BGR2GRAY)
l_raw = cv2.imread('left.jpg')
l_img = cv2.cvtColor(l_raw,cv2.COLOR_BGR2GRAY)
print('loading images')

#generating keypoints and descriptors for right and left images
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(r_img,None)
kp2, des2 = sift.detectAndCompute(l_img,None)
print('generating keypoints and descriptors')

#Using bfMatcher for matching correspondence between both images
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
print('generating correspondence with BFMatcher')

#filtering matches to obtain the best ones
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches = np.asarray(good)

#generating homography matrix
if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print('generating homography matrix')
#print H
else:
    raise AssertionError('Can’t find enough keypoints.')

#stitching images together
dst = cv2.warpPerspective(r_raw,H,(l_raw.shape[1] + r_raw.shape[1], l_raw.shape[0]))
print('Stitching images together')
plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:min(l_raw.shape[0],r_raw.shape[0]), 0:min(l_raw.shape[1],r_raw.shape[1])] = l_raw
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()

print('output Image saved')
