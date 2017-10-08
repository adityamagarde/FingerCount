# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 21:14:08 2017

@author: ADITYA
"""

import cv2
import numpy as np
import time
import csv

innerCircle = cv2.imread(r'Images/innerCircle.jpg')
innerCircle = cv2.cvtColor(innerCircle, cv2.COLOR_BGR2GRAY)

outerCircle = cv2.imread(r'Images/outerCircle.jpg')
outerCircle = cv2.cvtColor(outerCircle, cv2.COLOR_BGR2GRAY)

i = 0

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    rows, cols, channels = frame.shape
    rowsX = int(rows/2)
    colsX = int(cols/2) - int(cols/6)
    cropFrame = frame[0:rowsX, 0:colsX]
    grayFrame = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2HSV)
    lowerTone = np.array([0,30,10])
    upperTone = np.array([255,255,255])
    mask = cv2.inRange(hsv, lowerTone, upperTone)
    hsvFrame = cv2.bitwise_and(cropFrame, cropFrame, mask=mask)

    cv2.rectangle(frame, (0,0), (rowsX, colsX), (60,0,0), 2)
    cv2.circle(frame, (int(rowsX/2), int(colsX/2)), int(rowsX/4), (255,255,255))
    cv2.circle(frame, (int(rowsX/2), int(colsX/2)), int(rowsX/3), (255,255,255))

    kernel = np.ones((5,5), np.float32)/25
    smoothedFrame = cv2.filter2D(hsvFrame, -1, kernel)
    smoothedFrame = cv2.cvtColor(smoothedFrame, cv2.COLOR_HSV2BGR)
    smoothedFrame = cv2.cvtColor(smoothedFrame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(hsvFrame, (15,15), 2)
    ret, thresholdFrame = cv2.threshold(smoothedFrame, 70, 255, cv2.THRESH_BINARY)


#    medianCompoundFrame = cv2.medianBlur(medianFrame, 10)
#    cv2.imshow('threshold', thresholdFrame)
#    cv2.imshow('blur', blurFrame)
#    cv2.imshow('smoothed', smoothedFrame)
#    cv2.imshow('median', medianFrame)
#    cv2.imshow('edges', edges)

    andInner = cv2.bitwise_and(thresholdFrame, innerCircle)
    andOuter = cv2.bitwise_and(thresholdFrame, outerCircle)

    cv2.imshow('frame', frame)
    cv2.imshow('hsvFiltered', hsvFrame)
    cv2.imshow('threshold', thresholdFrame)
    cv2.imshow('andInner', andInner)
    cv2.imshow('andOuter', andOuter)

#    CREATING THE DATASET
    innerCircleValue = np.count_nonzero(andInner)
    outerCircleValue = np.count_nonzero(andOuter)
    area = np.count_nonzero(thresholdFrame)/51360

    print('inner', innerCircleValue)
    print('outer', outerCircleValue)
    print('area', area)
    i = i+1
    print('i',i)
    rowVariable = (innerCircleValue, outerCircleValue, area)

    with open(r'F:\Files\Projects\FingerCount\New folder\new.csv', 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(rowVariable)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    time.sleep(0.1)

cv2.destroyAllWindows()
cap.release()
