import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


# vid = cv2.VideoCapture("Voyager_3D_sbs.wmv")
# vid.grab()
# retval, color_image = vid.retrieve()
# height , width , layers =  color_image.shape
# full_image = cv2.cvtColor( color_image, cv2.COLOR_BGR2GRAY )
# new_h=int(height/4)
# new_w=int(width/4)
# image = cv2.resize(full_image, (new_w, new_h)) 

# height, width = image.shape[:2]
# start_row, start_col = int(0), int(0)
# end_row, end_col = int(height), int(width * .5)
# gray_l = image[start_row:end_row , start_col:end_col]
# gray_r = image[start_row:end_row , end_col:width]


gray_l = cv2.imread('iyari01l.jpg', 0)
gray_r = cv2.imread('iyari01r.jpg', 0)

# image = cv2.imread('historic-house-r.jpg', 0)
# height, width = image.shape[:2]
# print(image.shape)

# start_row, start_col = int(0), int(0)
# end_row, end_col = int(height), int(width * .5)
# gray_r = image[start_row:end_row , start_col:end_col]
# gray_l = image[start_row:end_row , end_col:width]

cv2.imshow('frame_l', gray_l)
cv2.imshow('frame_r', gray_r)

# SGBM Parameters -----------------
window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
 
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
 
# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
 
print('computing disparity...')
displ = left_matcher.compute(gray_l, gray_r)  # .astype(np.float32)/16
dispr = right_matcher.compute(gray_r, gray_l)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, gray_l, None, dispr)  # important to put "gray_l" here!!!
 
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imshow('Disparity Map', filteredImg)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


# stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
# disparity = stereo.compute(gray_l,gray_r)
# plt.imshow(disparity,'gray')
# plt.show()

# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.waitKey(0) 
# cv2.destroyAllWindows()




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


