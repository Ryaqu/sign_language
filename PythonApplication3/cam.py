import cv2
import numpy as np
import os
import imutils
import time

from collections import Counter

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import cnn_sgn

IMG_SIZE = 96
LR = 1e-3

nb_classes=28

MODEL_NAME = 'handsign.model'



model=cnn_sgn.cnn_model()

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
           
out_label=['A', 'B', 'C', 'D', 'E', 'empty' ]


pre=[]

s=''
cchar=[0,0]
c1=''

aWeight = 0.5
camera = cv2.VideoCapture(0)

top, right, bottom, left = 100, 150, 400, 450
num_frames = 0

flag=0
flag1=0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3));

while(True):
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)

    clone = frame.copy()
    (height, width) = frame.shape[:2]
    roi = frame[top:bottom, right:left]
    
    #canny
    #fgmask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);
    #fgmask = cv2.resize(fgmask, (IMG_SIZE,IMG_SIZE))
    #fgmask = cv2.Canny(fgmask, 50, 350)
    #img = fgmask  
   
    #grayScale
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #cv2.imshow("Video Feed 2", gray)
    #img = gray

    #sobel
    #img_gaussian = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #img_gaussian = cv2.GaussianBlur(img_gaussian,(3,3),0)
    #img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    #img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    #img = img_sobelx + img_sobely

    #pewitt
    img_gaussian = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(img_gaussian,(3,3),0)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    img = img_prewittx + img_prewitty
    
      
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

    cv2.imshow("Sobel", img)
    test_data = img

    orig = img
    data = img.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    pnb=np.argmax(model_out)
    print(str(np.argmax(model_out))+" "+str(out_label[pnb]))

    pre.append(out_label[pnb]) 
    cv2.putText(clone,
           '%s ' % (str(out_label[pnb])),
           (450, 150), cv2.FONT_HERSHEY_PLAIN,5,(255, 0, 0))

    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (255,0,0), 2)

    cv2.putText(clone,
                   '%s ' % (str(s)),
                   (10, 60), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 0))

    num_frames += 1

    cv2.imshow("Video Feed", clone)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
