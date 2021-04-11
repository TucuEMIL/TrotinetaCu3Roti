import cv2
import numpy as np
def resize(a,frame):
    scale_percent = a
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized
def putImage(x,y,name,dst):
    img=name
    width=40
    height=40
    dim=(width,height)
    #img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #imgimg=resize(a,img)
    height, width, channels = img.shape
    offset = np.array((y, x)) #top-left point from which to insert the smallest image. height first, from the top of the window
    dst[offset[0]:offset[0] + height, offset[1]:offset[1] + width] = img
