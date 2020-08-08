import os
import cv2
import numpy as np
from PIL import  Image
import pickle

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
imge_dir=os.path.join(BASE_DIR,"image")

face_Cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()

currentId=0
labale_ids={}
y_lables=[]
x_train=[]

for root,dirs,files in os.walk(imge_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            lable=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print(lable,path)
            if not lable in labale_ids:
                labale_ids[lable]=currentId
                currentId+=1

            id_=labale_ids[lable]
            print(labale_ids)
            #y_lables.append(lable) #some number
            #x_train.append(path) #verify this image , turn into a numpy array
            pil_image=Image.open(path).convert("L") #grayscale
           # size=(550,550)
           # final_image=pil_image.resize(size, Image.ANTIALIAS)
            image_array=np.array(pil_image,"uint8")
            print(image_array)
            faces=face_Cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_lables.append(id_)

#print(y_lables)
#print(x_train)
with open("labels.pickle", 'wb') as f:
    pickle.dump(labale_ids,f)
recognizer.train(x_train,np.array(y_lables))
recognizer.save("trainner.yml")