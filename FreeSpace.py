import time
import math
from extra import resize,putImage
import cv2
import numpy as np
from numpy import matrix
import imutils
pozition=0
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
def distance(p1,p2):
    dist = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return dist
def Check_frame(src):
    return src.shape[0],src.shape[1]
def Prepair_output(w,h):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out'+str(time.time())+'.avi', fourcc, 30.0, (w,h))
    return out
HeightList=[]
def Count_Groups(List):
    a=0
    for i in range(len(List)):
        if List[i]!=0 and List[i]>3:
            a=a+1
    return a
def FSD(src,window,segments):
    
    #afig = np.array( [[[0,0]]], dtype=np.int8 )*(segments+2)
    intersection=0
    #gradient=cv2.imread('grad.png',0)
    #cv2.imshow("grad",gradient)
    #road_line="line_road.png"
    R_list=[0]*(int)(segments/2)
    Max_reds=[0]*(int)(segments/2)
    HeightListRed=[0]*segments
    y_form=[10]*segments
    x_form=[0]*segments
    danger=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    #kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    GxR=0
    GyR=0
    R=0
    height = window.shape[0]
    width = window.shape[1]
    row=width
    x_cil=(int)(row/2)
    y_cil=(int)(row/2)
    l=0
    ok_out=0
    contorG=0
    contorR=0
    G_count=0
    R_count=0
    pic=np.array(src)
    #add_weight=src.copy()
    StangaLim=int(width/2)
    DreaptaLim=int(width/2)
    for i in range(int(width/2),width-10):
        if  pic[height-10][i]:
            DreaptaLim=i
            break
    
    for i in range(int(width/2)):
        if pic[height-10][int(width/2)-i]:
            StangaLim=int(width/2)-i
            break
    if(StangaLim==int(width/2)):
        StangaLim=1
    if(DreaptaLim==int(width/2)):
        DreaptaLim=width-1
    cv2.rectangle(window,(StangaLim,0),(90+StangaLim,40),(255,255,255),-1)
    cv2.rectangle(window,(StangaLim,0),(90+StangaLim,40),(0,0,0),2)
    #cv2.rectangle(window,(DreaptaLim-90,0),(DreaptaLim,40),(255,255,255),-1)
    #cv2.rectangle(window,(DreaptaLim-90,0),(DreaptaLim,40),(0,0,0),2)
    #cv2.rectangle(window,(tangaLim,0),(90+StangaLim,40),(0,0,0),2)
    cv2.circle(window,(StangaLim,height-10),10,(0,0,0),-1)    
    cv2.circle(window,(DreaptaLim,height-10),10,(0,0,0),-1)
    cv2.circle(window,(StangaLim,height-10),10,(255,255,255),3)
    cv2.circle(window,(DreaptaLim,height-10),10,(255,255,255),3)
    limitaMin=min(StangaLim,DreaptaLim-int(width/2))
    
    mijloc=int((DreaptaLim+StangaLim)/2)
    cv2.line(window,(DreaptaLim,height-10),(DreaptaLim,0),(255,255,255),3)
    cv2.line(window,(StangaLim,height-10),(StangaLim,0),(255,255,255),3)
    cv2.line(window,(DreaptaLim,height-10),(DreaptaLim,0),(0,0,0),2)
    cv2.line(window,(StangaLim,height-10),(StangaLim,0),(0,0,0),2)
    #segments=int((width/2-limita)/4)]
    for i in range(10):
        if int(mijloc-i*segments)+2*i*segments<DreaptaLim and int(mijloc-i*segments)+2*i*segments> StangaLim:
            tensor=i
    for i in range(segments):
       
            
       y_form[i]=int(mijloc-tensor*segments)+2*tensor*i
       
    for i in range(40,height-10):
        for j in range(segments):
            if pic[i][y_form[j]]: #and c1>i:
                        if x_form[j]<i:
                            x_form[j]=i
   # for i in range(1,segments+1):
    	#afig[i]=[y_form[i-1],x_form[i-1]]
    FreeSpaceZone = np.zeros([height,width,1],dtype=np.uint8)
   # a = np.array[(y_form[0],height)]
    for j in range(segments):
        #if j>1 and j<segments-1 and x_form[j]<x_form[j+1]+30 and x_form[j]<x_form[j-1]:
    	    #print("linie dubla"+str(time.time()))
        if x_form[j]>(height/2):
           
            contorR=contorR+1
            #cv2.line(window,(y_form[j],x_form[j]),(y_form[j],x_form[j]-10),(0,0,255),2)
            #cv2.line(window,(y_form[j],x_form[j]),(y_form[j],height),(0,0,255),1)
            
            #cv2.circle(FreeSpaceZone,(y_form[j],x_form[j]),4,(255),1)
            if j<segments-1 :#: and x_form[j+1]>(height/2):
                GyR=GyR+x_form[j]
                #cv2.line(window,(y_form[j+1],x_form[j+1]-10),(y_form[j],x_form[j]-10),(0,0,255),3)
                #cv2.line(src,(y_form[j+1],x_form[j+1]-10),(y_form[j],x_form[j]),(255),3)
                cv2.line(FreeSpaceZone,(y_form[j+1],x_form[j+1]),(y_form[j],x_form[j]),(255),1)
                
                GxR=GxR+y_form[j]
                R=R+1
            else:

                if(R>0):
                    #if(j<segments-1):
                        #cv2.line(FreeSpaceZone,(y_form[j+1],x_form[j+1]),(y_form[j],x_form[j]),(255),3)
                    road_line=window[int(GyR/R)-20:int(GyR/R)+20, int(GxR/R-20):int(GxR/R+20)]
                    #putImage(int(GxR/R-55),int(GyR/R)-95,road_line,window)
                    #cv2.line(window,(int(GxR/R),int(GyR/R)),(int(GxR/R),int(GyR/R)-50),(0,0,0),2)
                    #cv2.line(window,(int(GxR/R),int(GyR/R)),(int(GxR/R-60),int(GyR/R)-50),(255,200,200),1)### centru
                    #cv2.line(window,(int(GxR/R),int(GyR/R)),(int(GxR/R+60),int(GyR/R)-50),(255,200,200),1)### centru
                    #cv2.rectangle(window,(int(GxR/R-60),int(GyR/R)-50),(int(GxR/R)+60,int(GyR/R)-100),(255,200,200),1)
                    
                    #cv2.line(window,(y_form[j],x_form[j]),(y_form[j],x_form[j]-10),(0,0,255),2)
                    #cv2.circle(window,(y_form[30],x_form[30]-20),3,(255,0,0),1)
                    if R>=segments-1:
                        #b,g,r=window[x_form[30],y_form[30]-20]
                        #print(str(b)+"     "+str(g)+"     "+str(r))
                        #if ((b+g+r)/3>30):
    #print("the pixel is not black")
                                   
                        cv2.rectangle(window,(int(GxR/R-100),int(GyR/R)-105),(int(GxR/R)+270,int(GyR/R)-45),(255,255,255),-1)
                        cv2.rectangle(window,(int(GxR/R-100),int(GyR/R)-105),(int(GxR/R)+270,int(GyR/R)-45),(0,0,0),2)
                        #cv2.rectangle(window,(XL,YL-15),(XL+180,YL-40),(0,0,0),2)
                        #img4=cv2.imread("23.png")
                        #resized1=cv2.resize(img4,(60,60))
                        #putImage(int(GxR/R-160),int(GyR/R)-110,resized1,window)
                        cv2.putText(window," INTERSECTION ENTRY POINT ",(int(GxR/R-100),int(GyR/R)-70),font, 0.8,(0,0,0),2,cv2.LINE_AA)
                        intersection=1
                    #else:
                        #cv2.putText(window," ROAD LINE ",(int(GxR/R-10),int(GyR/R)-70),font, 0.4,(255,0,0),2,cv2.LINE_AA)
                    
                    
                    
                R=0
                GxR=0
                GyR=0
            
        else:
            contorG=contorG+1
            x_cil=x_cil+y_form[j]
            y_cil=y_cil+x_form[j]
           # cv2.circle(FreeSpaceZone,(y_form[j],x_form[j]),4,(255),1)
            #cv2.line(window,(y_form[j],x_form[j]),(y_form[j],x_form[j]-10),(0,255,0),2)
            #cv2.line(window,(y_form[j],x_form[j]),(y_form[j],height),(0,255,0),1)
            if j<segments-1:# and x_form[j+1]<(4*height/5):
                #cv2.line(window,(y_form[j+1],x_form[j+1]-10),(y_form[j],x_form[j]-10),(0,255,0),2)
                cv2.line(FreeSpaceZone,(y_form[j+1],x_form[j+1]),(y_form[j],x_form[j]),(255),1)
    cv2.line(FreeSpaceZone,(y_form[0],x_form[0]),(mijloc,height-10),(255),1)
    
    cv2.line(FreeSpaceZone,(y_form[segments-1],x_form[segments-1]),(mijloc,height-10),(255),1)
    #cv2.line(FreeSpaceZone,(StangaLim+1,height-10),(DreaptaLim-1,height-10),(255),1)
    #contours, hierarchy = cv2.findContours(FreeSpaceZone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("im2",contours)
    cnts = cv2.findContours(FreeSpaceZone, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(FreeSpaceZone, [c], 0, (255), -1)
    #cv2.drawContours(window, [c], 0, (0,0,0), 5)
    img=cv2.imread("grad1.jpg")
    mask_inv = cv2.bitwise_not(FreeSpaceZone)
    resiz=cv2.resize(img,(width,height),cv2.INTER_AREA)
    img2_fg = cv2.bitwise_and(resiz,resiz,mask = FreeSpaceZone)
    img1_bg = cv2.bitwise_and(window,window,mask = mask_inv)
    window = cv2.add(img1_bg,img2_fg)
    #cv2.imshow("1212",out_img)
    
    #mask_inv = cv2.bitwise_not(FreeSpaceZone)
    #foreground=resiz
    #background=FreeSpaceZone
    #FreeSpaceZone = cv2.resize(FreeSpaceZone, resiz.shape[1::-1])
   # FreeSpaceZone = FreeSpaceZone / 255
    #dst = cv2.bitwise_and(, FreeSpaceZone)
   #w dst = resiz  FreeSpaceZone
  #  cv2.imshow("dst",dst)
    #print(resiz.dtype)
    #sr1=np.array(FreeSpaceZone)
    #sr2=np.array(resiz)
    #dst = cv2.bitwise_and(sr2, sr1)
   # cv2.imshow("1",resiz)
    okL=0
    okR=0
    XR=0
    YR=0
    XL=0
    YL=0
    if intersection==0 :


          
        
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        #extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        if distance(extTop,extRight)>200 and distance(extTop,extLeft)>100 and extRight[1]<height-20 and extLeft[1]<height-20:
            
            if x_form[segments-1]<height/2 and x_form[segments-2]<height/2:
                XR=extRight[0]
                YR=extRight[1]
                okR=1
        #cv2.circle(window, (XR,YR), 8, (0, 0, 255), -1)
        #cv2.line(window,(XR,YR),(mijloc,height),(255,255,0),4)
                
                #v2.rectangle(window,(XL-10,YL-15),(XL+180,YL-50),(255,255,255),-1)
                #cv2.rectangle(window,(XL-10,YL-15),(XL+180,YL-50),(0,0,0),2)
                #cv2.putText(window,"LEFT ENTRY",(XL,YL-20),font, 0.8,(0,0,0),2,cv2.LINE_AA)
                
                

                
        #cv2.rectangle(window,(XR,YR),(XR+10,YR-10),(0,0,0),-1)
            if x_form[0]<height/2 and x_form[1]<height/2 and extLeft[0]+100<extTop[0]:
                okL=1
                XL=extLeft[0]
                YL=extLeft[1]
                #cv2.line(window,(XL,YL),(mijloc,height),(0,0,255),4)
                
            
                
            
        #cv2.circle(window, (XL,YL), 8, (255, 255, 0), -1)
        if(contorG>0.70*segments/2 and extTop[1]<height/3):
            #if contorG!=0:
            XT=extTop[0]
            YT=extTop[1]
            A=[XT,YT]
            #cv2.circle(window, (extTop[0],extTop[1]), 8, (255, 0, 255), -1)
            #cv2.line(window,(extTop[0],extTop[1]),(mijloc,height),(255,0,255),3)
            
            cv2.rectangle(window,(XT-10,YT-15),(XT+150,YT-50),(255,255,255),-11)
            cv2.rectangle(window,(XT-10,YT-15),(XT+150,YT-50),(0,0,0),2)
            cv2.putText(window,"TOP ENTRY",(XT,YT-25),font, 0.8,(0,0,0),2,cv2.LINE_AA)
        
        else:
            
            X_arrow=int(x_cil/(contorG+1))
            Y_arrow=int(y_cil/(contorG+1))
            A=[X_arrow,Y_arrow]
            #cv2.line(window,(X_arrow,Y_arrow),(mijloc,height),(255,255,255),3)
            #cv2.line(window,(X_arrow+20,Y_arrow),(X_arrow-20,Y_arrow),(255,255,255),3)
            #c#v2.line(window(int(x_cil/(contorG+1))+20,int(y_cil/(contorG+1))),(int(x_cil/(contorG+1))-20,int(y_cil/(contorG+1))),(255,255,255),3)
        
    cv2.line(FreeSpaceZone,(y_form[0],x_form[0]),(StangaLim,height-10),(255),1)
    cv2.line(FreeSpaceZone,(y_form[segments-1],x_form[segments-1]),(DreaptaLim,height-10),(255),1)
    cv2.line(FreeSpaceZone,(StangaLim+1,height-10),(DreaptaLim-1,height-10),(255),1)
    cnts = cv2.findContours(FreeSpaceZone, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(FreeSpaceZone, [c], 0, (255), -1)
    cv2.drawContours(window, [c], 0, (0,0,0), 7)
    img=cv2.imread("grad1.jpg")
    mask_inv = cv2.bitwise_not(FreeSpaceZone)
    resiz=cv2.resize(img,(width,height),cv2.INTER_AREA)
    img2_fg = cv2.bitwise_and(resiz,resiz,mask = FreeSpaceZone)
    img1_bg = cv2.bitwise_and(window,window,mask = mask_inv)
    window = cv2.add(img1_bg,img2_fg)
    if okR==1 or okL==1:
            if okR==1:
                cv2.rectangle(window,(XR-110,YR+20),(XR+70,YR-30),(255,255,255),-1)
                cv2.rectangle(window,(XR-110,YR+20),(XR+70,YR-30),(0,0,0),2)
                cv2.putText(window,"RIGHT ENTRY",(XR-100,YR),font, 0.8,(0,0,0),2,cv2.LINE_AA)
                
            if okL==1:
                cv2.rectangle(window,(XL-10,YL-15),(XL+180,YL-50),(255,255,255),-1)
                cv2.rectangle(window,(XL-10,YL-15),(XL+180,YL-50),(0,0,0),2)
                cv2.putText(window,"LEFT ENTRY",(XL,YL-20),font, 0.8,(0,0,0),2,cv2.LINE_AA)
            
            cv2.rectangle(window,(int(x_cil/(contorG+1)-110),height-90),(int(x_cil/(contorG+1)+180),height-130),(255,255,255),-1)
            cv2.rectangle(window,(int(x_cil/(contorG+1)-110),height-90),(int(x_cil/(contorG+1)+180),height-130),(0,0,0),2)
            cv2.putText(window,"INSIDE INTERSECTION",(int(x_cil/(contorG+1)-100),height-100),font, 0.8,(0,0,0),2,cv2.LINE_AA)
    #acceleration=
    #contours, hierarchy = cv2.findContours(FreeSpaceZone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("im2",contours)
    B=[mijloc,height]
    C=[width,height]
    #cv2.circle(window,(mijloc,height),50,(255,255,255),3) 
        
    angleD=0
    if intersection==0:
        angle=getAngle(A,B,C)
        cv2.line(window,(B[0]-10,B[1]),(A[0],A[1]),(255,255,255),3)
        cv2.line(window,(B[0]+10,B[1]),(A[0],A[1]),(255,255,255),3)
        contours = np.array([[B[0]-10,B[1]], [B[0]+10,B[1]], [A[0],A[1]]])
        cv2.fillPoly(window, pts = [contours], color =(0,0,0))
        cv2.circle(window,(mijloc,height),47,(255,255,255),3)
        if okR==1:
            contours = np.array([[B[0]-10,B[1]], [B[0]+10,B[1]], [XR,YR+100]])
            cv2.fillPoly(window, pts = [contours], color =(0,0,0))
            
            cv2.line(window,(B[0]-10,B[1]),(XR,YR+100),(255,255,255),2)
            cv2.line(window,(B[0]+10,B[1]),(XR,YR+100),(255,255,255),2)
            cv2.circle(window,(mijloc+50,height-10),25,(0,0,0),-1)
            cv2.circle(window,(mijloc+50,height-10),25,(255,255,255),2)
            angleD=getAngle((XL,YL),B,C)
            cv2.circle(window,(mijloc+50,height-10),25,(0,0,0),-1)
            cv2.circle(window,(mijloc+50,height-10),25,(255,255,255),2)
            cv2.putText(window,str(int(angleD)),(mijloc+45,height-10),font, 0.4,(255,255,255),1,cv2.LINE_AA)
        if okL==1:
            contours = np.array([[B[0]-60,B[1]-10], [B[0]-40,B[1]-10], [XL,YL+130]])
            #cv2.circle
            cv2.fillPoly(window, pts = [contours], color =(0,0,0))
            cv2.fillPoly(window, pts = [contours], color =(0,0,0))
            #cv2.circle(window,(mijloc,height),47,(0,0,0),3)
            cv2.line(window,(B[0]-60,B[1]-10),(XL,YL+130),(255,255,255),2)
            cv2.line(window,(B[0]-40,B[1]-10),(XL,YL+130),(255,255,255),2)
            
            angleL=getAngle((XL,YL),B,C)
            cv2.circle(window,(mijloc-50,height-10),25,(0,0,0),-1)
            cv2.circle(window,(mijloc-50,height-10),25,(255,255,255),2)
            cv2.putText(window,str(int(angleL)),(mijloc-70,height-10),font, 0.4,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(window,(mijloc,height),47,(0,0,0),-1)
       # cv2.circle(window,(mijloc,height),47,(255,255,255),-1)
        #
        if angle<100:
            cv2.putText(window,str(int(angle)),(mijloc-17,height-10),font, 0.8,(255,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(window,str(int(angle)),(mijloc-26,height-10),font, 0.8,(255,255,255),2,cv2.LINE_AA)
        #cv2.circle(window, extBot, 8, (255, 255, 0), -1)
        #print(pozition)
   # elif R>segments-2:
        #cv2.putText(window," INTERSECTION ENTRY POINT ",(int(GxR/R-100),int(GyR/R)-70),font, 0.8,(0,0,0),2,cv2.LINE_AA)
        #img3=cv2.imread("12.png")
        
        #resized=cv2.resize(img3,(40,40))
        
        #putImage(mijloc-20,height-42,resized,window)
    
    	#cv2.putText(window,"",(mijloc-17,height-10),font, 0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(window,"Width: "+str(width),(StangaLim+5,11),font, 0.4,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(window,"Height: "+str(height),(StangaLim+5,23),font, 0.4,(0,0,0),1,cv2.LINE_AA)
    
    #cv2.putText(menu,"Exit",(70,117),font, 0.4,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow("FreeSpace",FreeSpaceZone)
    
    
    
    
    
    
    #if(pozition>10)
       
    
    
    return window,intersection,StangaLim
