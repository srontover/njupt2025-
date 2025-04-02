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
    
    ut.follow_line(img_canny, HEIGHT, WIDTH, 500, 15)
    ut.get_signal(img_canny, HEIGHT, WIDTH, 15, 15)
    
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break