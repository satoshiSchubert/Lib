# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:00:07 2021
@author: HuangHongxiang
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

#>>>>>>>>>>Ques1>>>>>>>>>>>
def unpad(array,width):
    return array[width:-width,width:-width]

def sum_self(array):
    Sum = 0
    size = array.shape[0]
    for i in range(size):
        for j in range(size):
            Sum += array[i,j]
    return Sum

def erode(raw,kernel):
    imgSize=raw.shape[0]
    kerSize=kernel.shape[0]
    new = np.zeros((imgSize,imgSize))
    for i in range(imgSize-kerSize):
        for j in range(imgSize-kerSize):
            v = raw[i:i+kerSize,j:j+kerSize]*kernel
            if(sum_self(v)==sum_self(kernel)):
                new[i,j] = 1
    return new

def dilate(raw,kernel):
    imgSize=raw.shape[0]
    kerSize=kernel.shape[0]
    new = np.zeros((imgSize,imgSize))
    for i in range(imgSize-kerSize):
        for j in range(imgSize-kerSize):
            if(raw[i,j]==1):
                new[i:i+kerSize,j:j+kerSize] += kernel[:,:]
                
    for i in range(new.shape[0]):
        for j in range(new.shape[0]):
            new[i,j] = 1 if(new[i,j]>=1) else 0
    return new

def openOp(raw,kernel):
    return dilate(erode(raw,kernel),kernel)

def closeOp(raw,kernel):
    return erode(dilate(raw,kernel),kernel)

raw = np.zeros((6,6))
raw[1,1] = 1
raw[1,3] = 1
raw[2,1:4] = 1
raw[3,2:4] = 1
raw[4,2] = 1

kernel = np.zeros((6,6))
kernel[2,2:4] = 1
kernel[3,2] = 1
kernel = unpad(kernel,2)

fig, axarr = plt.subplots(3,2,figsize=(6,10))
axarr[0][0].imshow(raw)
axarr[0][0].title.set_text("raw image")
axarr[0][1].imshow(np.pad(kernel,2))
axarr[0][1].title.set_text("kernel")

axarr[1][0].title.set_text("eroded")
axarr[1][0].imshow(erode(raw,kernel))
axarr[1][1].title.set_text("dilated")
axarr[1][1].imshow(dilate(raw,kernel))
axarr[2][0].title.set_text("open operation")
axarr[2][0].imshow(openOp(raw,kernel))
axarr[2][1].title.set_text("close operation")
axarr[2][1].imshow(closeOp(raw,kernel))
#<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>Ques2>>>>>>>>>>>>
src = cv2.imread("Fig0911(a)(noisy_fingerprint).tif")
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel_6 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
dst = cv2.erode(src,kernel_2)
dst = cv2.dilate(dst,kernel_6)
dst = cv2.erode(src,kernel_2)
plt.figure()
plt.imshow(dst)
#<<<<<<<<<<<<<<<<<<<<<<<<<<




#<<<<<<<<<<<<<<<<<<<<<<<<<<



