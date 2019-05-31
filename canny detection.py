import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

lower_orange = np.array([5,100,100])
upper_orange = np.array([15,255,255])

while(True):
    ret,frame = cam.read()

    #Smoothen the image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

      
    #Threshold the image for orange color
    image_hsv = cv.cvtColor(image_smooth,cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(image_hsv,lower_orange,upper_orange)

                                          
    cv.imshow('Frame',image_threshold)
    if cv.waitKey(10) == 27:
        break

cam.release()
cv.destroyAllWindows()
