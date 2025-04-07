import cv2 as cv
import numpy as np
import time
import utilies as ut

HEIGHT = 480
WIDTH = 640

cap = cv.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while True:
    success, img = cap.read()
    img_copy = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 1)
    img_canny = cv.Canny(img_blur, 50, 50)
    cv.imshow("Canny", img_canny)
    
    follow_return = ut.follow_line(img_canny, img_copy, HEIGHT, WIDTH, 500, 15)
    signal_return = ut.get_signal(img_canny, img_copy, HEIGHT, WIDTH, 15, 15)
    print(follow_return)
    if signal_return != None:
        print(signal_return)
        adjust_return = ut.adjust_position(img_canny, img_copy, HEIGHT, WIDTH, 500)
        if adjust_return!= None:
            print(adjust_return)
    
    cv.imshow("Result", img_copy)
    
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()