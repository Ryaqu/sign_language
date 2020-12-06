import cv2
import numpy as np
import os

IMG_SIZE=96

top, right, bottom, left = 100, 150, 400, 450
exit_con='esc'
a=''
dir0='data'

try:
    os.mkdir(dir0)
except:
    print('Data folder already created.')

camera = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3));

while(True):

    a=input('Exit: esc or Enter the name of the gesture : ')

    if a==exit_con:
        break

    dir1=str(dir0)+'/'+str(a)

    try:
        os.mkdir(dir1)
    except:
        print('Folder already existing')

    i=0
    while(True):
        (t, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        roi = frame[top:bottom, right:left]

        #grayScale
        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        #gray = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))
        #cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),gray)
        #cv2.imshow("Video Feed 1", gray)

        #sobel
        #img_gaussian = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #img_gaussian = cv2.GaussianBlur(img_gaussian,(3,3),0)
        #img_gaussian = cv2.resize(img_gaussian, (IMG_SIZE,IMG_SIZE))
        #img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
        #img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
        #img = img_sobelx + img_sobely
        #cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),img)
        #cv2.imshow('frame',img)

        #prewitt
        img_gaussian = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(img_gaussian,(3,3),0)
        img_gaussian = cv2.resize(img_gaussian, (IMG_SIZE,IMG_SIZE))
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        img = img_prewittx + img_prewitty
        cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),img)
        cv2.imshow('frame',img)

        #canny
        #fgmask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);
        #fgmask = cv2.resize(fgmask, (IMG_SIZE, IMG_SIZE))
        #fgmask = cv2.Canny(fgmask, 50, 350)
        #cv2.imshow('frame', fgmask)
        #cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),fgmask)

        i += 1
        print(i)
        if i > 400:
            break

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        cv2.imshow("Video Feed", frame)
        keypress = cv2.waitKey(1)

        if keypress == 27:
            break

camera.release()
cv2.destroyAllWindows()
