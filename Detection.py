import cv2
import numpy as np
from extra import resize
from Compare import compare
def RED(img,frame):
	
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lower_red = np.array([0,120,70])
	upper_red = np.array([10,255,255])
	mask1 = cv2.inRange(hsv, lower_red, upper_red)
	lower_red = np.array([170,120,70])
	upper_red = np.array([180,255,255])
	mask2 = cv2.inRange(hsv,lower_red,upper_red)
	mask=mask1+mask2
	kernel = np.ones((10,10), np.uint8)
	img_dilation = cv2.dilate(mask, kernel, iterations=1)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_GRADIENT, kernel)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)
	edge=cv2.Canny(closing,120,100)
	contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(c) for c in contours]
	if areas[np.argmax(areas)]>500:
		max_index = np.argmax(areas)
		cnt=contours[max_index]
		x,y,w,h = cv2.boundingRect(cnt)
		crop_img = img[y:y+h, x:x+w]
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),1)
		cv2.rectangle(img,(x,y),(x+w,y+10),(255,255,0),-1)
		
		width = int(crop_img.shape[1])
		height = int(crop_img.shape[0])
		dim = (width, height)
		img_stop=cv2.imread("TRR.png")
		resized = cv2.resize(img_stop, dim, interpolation = cv2.INTER_AREA)
		a=compare(frame,resized,crop_img)
		#print(a)
		if a>2:
	    		cv2.putText(frame, 'Semafor', (x+int(0.66*frame.shape[1]),y+10), cv2.FONT_HERSHEY_SIMPLEX ,  0.3, (0,0,0), 1, cv2.LINE_AA)
		else:
            		img_stop=cv2.imread("1.jpeg")
            		resized = cv2.resize(img_stop, dim, interpolation = cv2.INTER_AREA)
            		a=compare(frame,resized,crop_img)
            		if a>2:
            			cv2.putText(frame, 'STOP', (x+int(0.66*frame.shape[1]),y+10), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.3, (0,0,0), 1, cv2.LINE_AA)
	#cv2.imshow("Resized image", resized)
	#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	#return img
def GREEN(img):
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	low_green = np.array([25, 52, 72])
	high_green = np.array([80, 255, 255])
	mask = cv2.inRange(hsv,low_green,high_green)
	
	kernel = np.ones((10,10), np.uint8)
	img_dilation = cv2.dilate(mask, kernel, iterations=1)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_GRADIENT, kernel)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)
	edge=cv2.Canny(closing,120,100)
	contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	
	#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	#return img
def BLUE(img):
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	low_green = np.array([94, 80, 2])
	high_green = np.array([126, 255, 255])
	mask = cv2.inRange(hsv,low_green,high_green)
	
	kernel = np.ones((10,10), np.uint8)
	img_dilation = cv2.dilate(mask, kernel, iterations=1)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_GRADIENT, kernel)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)
	edge=cv2.Canny(closing,120,100)
	contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]
	x,y,w,h = cv2.boundingRect(cnt)
	
	#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
	#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	#return img
def dt(img,frame):
#img=cv2.imread("1.png")
#ret,thresh1 = cv2.threshold(img,75,85,cv2.THRESH_BINARY)

#edge=cv2.Canny(img,50,10)
#cv2.imshow('image',img)
	try:
		RED(img,frame)
		#GREEN(img)
		#BLUE(img)
	except:
		pass
		#cv2.imshow('tr',GREEN(img))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
