import numpy as np
import cv2
from matplotlib import pyplot as plt

cimgL = cv2.imread('left.png', 0)
cimgR = cv2.imread('right.png', 0)
# imgL = cv2.cvtColor(cimgL, cv2.COLOR_BGR2GRAY )
# imgR = cv2.cvtColor(cimgR, cv2.COLOR_BGR2GRAY )
# plt.imshow(imgL)
# plt.show()

# stereo = cv2.StereoBM(1, 16, 15)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(cimgL,cimgR)
plt.imshow(disparity,'gray')
plt.show()


