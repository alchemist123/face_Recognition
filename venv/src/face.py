import cv2
import numpy as np

face_Cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0);

while True:

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_Cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        #recognize using Deep lerened model

        img_item="my-image.png"
        cv2.imwrite(img_item,roi_gray)
        end_cordx=x+w
        end_cordy=y+h
        color=(0,255,0)
        stroke=2
        cv2.rectangle(frame,(x,y),(end_cordx,end_cordy),color,stroke)


    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()