import cv2
import numpy as np

cap=cv2.VideoCapture(0)

def empty(a):
    pass

cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments",(300,300))
cv2.createTrackbar("Thresh", "Color Adjustments",0,255,empty)
cv2.createTrackbar("Hue Min","Color Adjustments",0,255,empty)
cv2.createTrackbar("Sat Min","Color Adjustments",0,255,empty)
cv2.createTrackbar("Val Min","Color Adjustments",0,255,empty)
cv2.createTrackbar("Hue Max","Color Adjustments",255,255,empty)
cv2.createTrackbar("Sat Max","Color Adjustments",255,255,empty)
cv2.createTrackbar("Val Max","Color Adjustments",255,255,empty)

while True:
    _,img = cap.read()
    img = cv2.flip(img,2)
    img = cv2.resize(img,(600,500))
    cv2.rectangle(img, (0,1), (300,500), (255, 0, 0), 0)
    crop_image = img[1:500, 0:300]

    img_hsv = cv2.cvtColor(crop_image,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","Color Adjustments")
    s_min = cv2.getTrackbarPos("Sat Min","Color Adjustments")
    v_min = cv2.getTrackbarPos("Val Min","Color Adjustments")
    h_max = cv2.getTrackbarPos("Hue Max","Color Adjustments")
    s_max = cv2.getTrackbarPos("Sat Max","Color Adjustments")
    v_max = cv2.getTrackbarPos("Val Max","Color Adjustments")
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(img_hsv,lower,upper)
    img_res = cv2.bitwise_and(crop_image, crop_image, mask=mask)
    mask1 = cv2.bitwise_not(mask)
    thresh_val = cv2.getTrackbarPos("Thresh","TrackBars")
    res,thres = cv2.threshold(mask1, thresh_val,255,cv2.THRESH_BINARY)
    dilata = cv2.dilate(thres,(3,3),iterations = 6)
    cnts,hier = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    cv2.imshow('Img',img)
    cv2.imshow('Mask',mask)
    cv2.imshow('Result',thres)

    if cv2.waitKey(1) and 0xFF==ord('q'):
        break
