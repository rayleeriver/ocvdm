import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# cap_l = cv2.VideoCapture(0)
# cap_r = cv2.VideoCapture(1)
# ret_l, frame_l = cap_l.read()
# ret_r, frame_r = cap_r.read()
# gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
# gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

gray_l = cv2.imread('im2.png', 0)
gray_r = cv2.imread('im6.png', 0)

# image = cv2.imread('historic-house-r.jpg', 0)
# height, width = image.shape[:2]
# print(image.shape)

# start_row, start_col = int(0), int(0)
# end_row, end_col = int(height), int(width * .5)
# gray_r = image[start_row:end_row , start_col:end_col]
# gray_l = image[start_row:end_row , end_col:width]

cv2.imshow('frame_l', gray_l)
cv2.imshow('frame_r', gray_r)

stereo = cv2.StereoBM_create(numDisparities=80, blockSize=15)
disparity = stereo.compute(gray_l,gray_r)
plt.imshow(disparity,'gray')
plt.show()

# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cv2.waitKey(0) 
cv2.destroyAllWindows()




# while(True):
    # ret_l, frame_l = cap_l.read()
    # ret_r, frame_r = cap_r.read()
    # gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    # gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame_l', gray_l)
    # cv2.imshow('frame_r', gray_r)

    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity = stereo.compute(gray_l,gray_r)
    # plt.imshow(disparity,'gray')
    # plt.show()

    # if cv2.waitKey(1):
    #     if 0xFF == ord('c'):
    #         break
    #     if 0xFF == ord('q'):
    #         sys.exit(0)




# cimgL = cv2.imread('left.png', 0)
# cimgR = cv2.imread('right.png', 0)
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(cimgL,cimgR)
# plt.imshow(disparity,'gray')
# plt.show()


