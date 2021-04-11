import os
from Detection import dt 
try:
    import cv2
    import numpy as np
    from numpy import matrix
    import imutils
    import time
    print("[Status]All Components are working properly... ")
except:
    print("[Status]Installing OpenCV & Numpy... ")
    os.system('cmd /c "py -m pip install opencv-python"')
    print("[Status]OpenCV                         INSTALLED")

    print("[Status]Numpy                          INSTALLED")
    print("[Status]Installing sklearn...")
    os.system('cmd /c "py -m pip install sklearn"')
    print("[Status]Sklearn                        INSTALLED")
import time
import math
from extra import resize,putImage
from FreeSpace import FSD,Check_frame,Prepair_output
#---------------------------------------------------
video_name="bfmc2020_online_1.avi"   #asta modifici pentru alt video
SAVE_VIDEO=False      #aici salvam video
crop=100               #cropuim ROI
dim=50                #rezolutila in %
segments=76          #numaru de segmente
#---------------------------------------------------


    #pass
capture=cv2.VideoCapture(video_name) ###pentru video  ((((Cu resolutia 640 pe 480)))
ret,frame=capture.read()
	
#---------------------------------------------------------------------
Check_frame(frame)
out=Prepair_output(820,416)
#---------------------------------------------------------------------
print("[Running] READY")
print("START !!!")
#---------------------------------------------------------------------------------------------------------
while(capture.isOpened()):
    #img=cv2.imread("grad.png")
    #cv2.imshow("img",img)
    timp_initial=time.time()
    ret,frame=capture.read()
    frame=resize(dim,frame)
    height,width = frame.shape[0],frame.shape[1]
    frame= frame[100:height-crop, 0:width]
    h,w = frame.shape[0],frame.shape[1]
    
    #StatusMenu = np.zeros([h,200,3],dtype=np.uint8)
    #StatusMenu.fill(150)
    edge=cv2.Canny(frame,120,150)
   # kernel = np.ones((10, 10), 'uint8')
    #edge=cv2.dilate(edge, kernel, iterations=1)
    #cv2.rectangle(frame,(0,0),(200,120),(0,0,0),-1)
   # cv2.rectangle(frame,(0,0),(200,20),(0,255,255),-1)
    #cv2.putText(frame,"Detection Activity ",(10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,0),1,cv2.LINE_AA)
    
    #dt(frame[0:h,int(w*0.66):w],frame)
    
    frame,det,stg=FSD(edge,frame,segments)
    #if det==1:
    
    #vis = np.concatenate((frame, StatusMenu), axis=1)
    #cv2.line(StatusMenu,(0,int(h/2)),(200,int(h/2)),(0,0,0),3)
    #cv2.putText(StatusMenu,"Resolution: "+str(dim)+" %",(10,int(height/2)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,255,255),1,cv2.LINE_AA)
    
    
    #vis = np.concatenate((frame, StatusMenu), axis=1)
    timpfinal=time.time()-timp_initial
    #print()
    cv2.putText(frame,"FPS: "+str(int(1/timpfinal)),(stg+5,35),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow("Menu",frame)
    
    #if SAVE_VIDEO==True:
    out.write(frame)
    cv2.imshow("Edge",edge)
#-----------------------------------------------------------------------------------------------------
    key = cv2.waitKey(1)
    if key == ord('q')or key==ord("Q"):
        break
    elif key== ord("a")or key==ord("A"):
        dim=dim-10
    elif key== ord("d")or key==ord("D"):
        dim=dim+10
    elif key==ord("r")or key==ord("R"):
        if SAVE_VIDEO==False:
            SAVE_VIDEO=True
        else:
            SAVE_VIDEO=False
    elif key==ord("z")or key==ord("Z"):
    	a=int(input())
        #segments=segments-2
    elif key==ord("c")or key==ord("C"):
        
        segments=segments+2
#-------------------------------------------------------------------------------------------------------        
capture.release()
cv2.destroyAllWindows()


