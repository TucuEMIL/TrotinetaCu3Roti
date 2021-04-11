import cv2
import sys
import os.path
import numpy as np
def putImage(x,y,img,dst):
    
    #img=name
    img=cv2.resize(img, (200,100), interpolation = cv2.INTER_AREA)
    height, width, channels = img.shape
    offset = np.array((y, x)) #top-left point from which to insert the smallest image. height first, from the top of the window
    dst[offset[0]:offset[0] + height, offset[1]:offset[1] + width] = img
def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)
        

    return out

def compare(src,img1, img2):
             # queryImage
            # trainImage

    # Initiate SIFT detector
    sift=cv2.xfeatures2d.SIFT_create()
    contor=0

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    good = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    for m,n in matches:
    	if m.distance <0.7*n.distance:
             good.append([m])
             contor=contor+1
    #matches = sorted(matches, key=lambda val: val.distance)
    #print(len(matches))
    #img3 = drawMatches(img1,kp1,img2,kp2,)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None,flags=2)
    # Show the image
    #img3=cv2.add(src,img2_fg)
    putImage(0,20,img3,src)
    #cv2.imshow('Matched Features', img3)
    return contor
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

#compare("1.jpeg","12.png")
