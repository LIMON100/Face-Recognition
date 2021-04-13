# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:21:41 2021

@author: limon
"""

import cv2
import numpy as np
import os 
import pickle


faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")


labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}



font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0


# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height


# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

p = 2

current_id = 0
label_ids = {}
y_labels = []
x_train = []


def train_dat():
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, 'new_dataset')
    
    
    
    for root, dirs, files in os.walk(image_dir):
        
        for file in files:
            
            if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png") or file.endswith("jfif"):
                
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]
                
                pil_image = Image.open(path).convert("L")
                size = (550, 550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")
                
                faces - faceCascade.detectMultiScale(image_array)
                
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    
                    
    with open("pickles/face-label.pickle", "wb") as f:
        pickle.dump(label_ids, f)
    

    recognizer.train(x_train, np.array(y_labels))
    recognized.write("recognizer/face-trainner.yml")




def make_dataset(img):
    
    count = 0
    p = 2
      
    dir_n = "new_dataset/person" + str(p)
    os.mkdir(dir_n)
    img_count = 1
    
    while(True):
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_detector.detectMultiScale(rgb, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
        
            
            #createFolder('./dataset/')
            # Save the captured image into the datasets folder
            
            #os.mkdir(pathn)
            #pathn = os.makedirs(os.path.join('subfolder' + str(p)))
            #cv2.imwrite(pathn + str(face_id) + '.' + str(count) + ".jpg", rgb[y:y+h,x:x+w])
            cv2.imwrite(dir_n + f"\image{img_count}.jpg", rgb[y:y+h,x:x+w])
    
            cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        img_count += 1
        
        if k == 27:
            break
        elif count >= 25: 
             break
    
    p = p + 1
    
    train_data()


while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x, y, w, h) in faces:
        
        flag = 1

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        #id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # Check confidence
        if conf > 4 and conf <= 85:
            #id = names[id]
            names = labels[id_]
            
            ##check if the face is in the dataset
            for key, value in labels.items():
                if names != value:
                    flag = 0
            
            ## If face not match then it take snapshot and tranining
            if flag == 0:
                make_dataset(img)
            
            else:
                cv2.putText(img, str(names), (x+5,y-5), font, 1, (255,255,255), 2)
        
        
        #cv2.putText(img, str(names), (x+5,y-5), font, 1, (255,255,255), 2)
        #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()