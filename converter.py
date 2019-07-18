import cv2
import numpy as np
import os


def locate_letters(image):
    
    output = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_cnt, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    letters = []

    (H,W) = image.shape[:2]

    for i, cnt in enumerate(cnts):

        if hierarchy[0][i][3] == 0:

            (x,y,w,h) = cv2.boundingRect(cnt)
            roi = thresh[y-5:y+h+5, x-5:x+w+5].copy()

            hull = cv2.convexHull(cnts[i], False)
            mask = np.zeros((H,W), dtype=np.uint8)
            mask = cv2.drawContours(mask, [hull], -1, (255,255,255), -1)
            mask = mask[y-5:y+h+5, x-5:x+w+5].copy()

            boxes.append((y-5, y+h+5, x-5, x+w+5))

            roi[mask == 0] = 255 
            letters.append(roi)
            
            
    return letters, boxes


def preprocess_letters(letters):
    
    for i, letter in enumerate(letters):
        
        roi = letter
        
        """scale_percent = 150
        width = int(roi.shape[1] * scale_percent / 100)
        height = int(roi.shape[0] * scale_percent / 100)
        dim = (width, height)

        roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)"""
        blurred = cv2.GaussianBlur(roi, (7,7), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        letters[i] = thresh
    
    return letters


def affect_letters(letters, mode, ksize=5, iterations=3):
    
    for i, letter in enumerate(letters):

        if mode == 'erode':
            eroded = cv2.erode(letter, (ksize, ksize), iterations=iterations)
            letters[i] = eroded
        elif mode == 'dilate':
            dilated = cv2.dilate(letter, (ksize, ksize), iterations=iterations)
            letters[i] = dilated

        #cv2.imwrite('Result/{}.png'.format(i), eroded)
        
    return letters