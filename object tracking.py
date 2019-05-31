import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

lower_orange = np.array([5,100,100])
upper_orange = np.array([15,255,255])

while(True):
    ret,frame = cam.read()
    frame = cv.flip(frame,1)

    #Smoothen the image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

      
    #Threshold the image for orange color
    image_hsv = cv.cvtColor(image_smooth,cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(image_hsv,lower_orange,upper_orange)

    # Find contours
    contours,hierarchy = cv.findContours(image_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

    # Find the index of the largest contour
    if(len(contours)!=0):
        area = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(area)
        cnt = contours[max_index]
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                                          
    cv.imshow('Frame',frame)
    if cv.waitKey(10) == 27:
        break

cam.release()
cv.destroyAllWindows()
