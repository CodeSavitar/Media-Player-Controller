import cv2
import numpy as np
import pyautogui as pag
import math

cap=cv2.VideoCapture(0)

colors = [29,0,0,225,255,180]

def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",(300,300))
cv2.createTrackbar("Thresh", "TrackBars",0,255,empty)
cv2.createTrackbar("Hue_Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue_Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat_Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat_Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val_Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val_Max","TrackBars",255,255,empty)

while True:
    _,img = cap.read()
    img = cv2.flip(img,2)
    img = cv2.resize(img,(600,500))
    cv2.rectangle(img, (0,1), (300,500), (255, 0, 0), 0)
    crop_image = img[1:500, 0:300]

    img_hsv = cv2.cvtColor(crop_image,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue_Min","Trackbars")
    h_max = cv2.getTrackbarPos("Hue_Max","Trackbars")
    s_min = cv2.getTrackbarPos("Sat_Min","Trackbars")
    s_max = cv2.getTrackbarPos("Sat_Max","Trackbars")
    v_min = cv2.getTrackbarPos("Val_Min","Trackbars")
    v_max = cv2.getTrackbarPos("Val_Max","Trackbars")
    lower = np.array(colors[0:3])
    upper = np.array(colors[3:6])
    mask = cv2.inRange(img_hsv,lower,upper)
    img_res = cv2.bitwise_and(crop_image, crop_image, mask=mask)
    mask1 = cv2.bitwise_not(mask)
    thresh_val = cv2.getTrackbarPos("Thresh","TrackBars")
    res,thres = cv2.threshold(mask1, thresh_val,255,cv2.THRESH_BINARY)
    dilata = cv2.dilate(thres,(3,3),iterations = 6)

    cnts,hier = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    try:

        cont_max = max(cnts, key=lambda x: cv2.contourArea(x))
        epsilon = 0.0005*cv2.arcLength(cont_max,True)
        approx = cv2.approxPolyDP(cont_max,epsilon,True)
        hull = cv2.convexHull(cont_max)

        cv2.drawContours(crop_image, [cont_max], -1, (50, 50, 150), 2)
        cv2.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

        hull = cv2.convexHull(cont_max, returnPoints=False)
        defects = cv2.convexityDefects(cont_max, hull)
        count_defects = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cont_max[s][0])
            end = tuple(cont_max[e][0])
            far = tuple(cont_max[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            if angle <= 50:
                count_defects += 1
                cv2.circle(crop_image,far,5,[255,255,255],-1)

        print("count==",count_defects)

        if count_defects == 0:
            
            cv2.putText(img, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2)

        elif count_defects == 1:
            
            pag.press("space")
            cv2.putText(img, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2)

        elif count_defects == 2:

            pag.press("down")           
            cv2.putText(img, "Volume Down", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2)

        elif count_defects == 3:

            pag.press("up")           
            cv2.putText(img, "Volume Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2)

        elif count_defects == 4:

            pag.press("right")
            cv2.putText(img, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2)

        elif count_defects == 4:

            pag.press("left")
            cv2.putText(img, "Rewind", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2)    

        else:
            pass  

    except:
        pass          


    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cap.destroyAllWindows()    
    
