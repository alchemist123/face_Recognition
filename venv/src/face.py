import cv2
import numpy as np
import  pickle

face_Cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
eyes_Cascade=cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
og_labels={"person_man":0}
with open("labels.pickle", 'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0);

while True:

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_Cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        id_, conf=recognizer.predict(roi_gray)
        if conf>=37 :#and conf<=85:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,55,255)  #bgr(102, 255, 51)
            stroke1=1
            cv2.putText(frame,name,(x,y),font,1,color,stroke1,cv2.LINE_AA)
        #recognize using Deep lerened model

        img_item="my-image.png"
        cv2.imwrite(img_item,roi_gray)
        end_cordx=x+w
        end_cordy=y+h
        color=(0,255,0)
        stroke=1
        cv2.rectangle(frame,(x,y),(end_cordx,end_cordy),color,stroke)
        eyes=eyes_Cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)


    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()