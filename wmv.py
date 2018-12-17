import cv2
vid = cv2.VideoCapture("Voyager_3D_sbs.wmv")

while True:
    vid.grab()
    retval, color_image = vid.retrieve()
    if not retval:
        break

    height , width , layers =  color_image.shape

    image = cv2.cvtColor( color_image, cv2.COLOR_BGR2GRAY )
    new_h=int(height/4)
    new_w=int(width/4)
    resize = cv2.resize(image, (new_w, new_h)) 

    cv2.imshow("Test", resize)
    cv2.waitKey(1)