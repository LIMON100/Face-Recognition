# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:21:41 2021

@author: limon
"""

import cv2
import numpy as np
from PIL import Image
import os 
import pickle


faceCascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/cascades/data/haarcascade_eye.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("./recognizers/face-trainner.yml")


font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0
global p
p = 0
save_value = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height


# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


current_id = 0
label_ids = {}
y_labels = []
x_train = []
a = []

a.append(p)

f= open("ltest.txt","a")

f.write(str(p))
f.write("\n")
f.close()


def train_data():
    
    print("Start Trining...............................")
    
    global current_id
    global label_ids
    global y_labels
    global x_train
    global a
    
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
                
                print("Collect Face............")
                faces - faceCascade.detectMultiScale(image_array)
                
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    
    
    print("Saving class names.......................")
    print("\n")
    with open("pickles/face-labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)
    
    
    print("Save file to yml..........................")
    print("\n")
    recognizer.train(x_train, np.array(y_labels))
    recognizer.write("recognizers/face-trainner.yml")
    
    print("Finish Training--------------------------")
    print("\n")




def make_dataset(img):
    
    
    print("Make New Dataset.......................")
    
    #p = q
    count = 0
    global p
    global save_value
    
    #a.append(p)
    b = a[-1] + 12
    #b = p
    #print(b)
      
    dir_n = "I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/new_dataset/person" + str(b)
    os.mkdir(dir_n)
    img_count = 1
    
    with open("ltest.txt", "r") as fp:
        lines = fp.readlines()
        end = lines[-1].split(',')[0]
    
    save_value = end
    
    while(True):
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(rgb, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            
            #os.mkdir(pathn)
            #pathn = os.makedirs(os.path.join('subfolder' + str(p)))
            #cv2.imwrite(pathn + str(face_id) + '.' + str(count) + ".jpg", rgb[y:y+h,x:x+w])
            cv2.imwrite(dir_n + f"\image{img_count}.jpg", rgb[y:y+h,x:x+w])
    
            cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        img_count += 1
        #p = p + 1
        
        if k == 27:
            break
        elif count >= 25:
            p = p + 1
            a.append(p)
            
            f = open("ltest.txt", "a")
            save_value = int(save_value) + 1
            f.write(str(save_value))
            f.write("\n")
            f.close()
            break
    
    
    print("Complete Making Dataset.....................")
    #cam.release()
    #a.append(p)
    #p = p + 1
    #q = p
    #a.append(p)
    print("Goes to Training")
    print("\n")
    #cam.release()
    #cv2.destroyAllWindows()
    train_data()


while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #flag = 1
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
       )
    
    recognizer.read("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/recognizers/face-trainner.yml")


    labels = {"person_name": 1}
    with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/pickles/face-labels.pickle", 'rb') as f:
    	og_labels = pickle.load(f)
    	labels = {v:k for k,v in og_labels.items()}

    
    #cv2.imshow("img2", img)
    #flag = 0
    for(x, y, w, h) in faces:
        
        flag = 0
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        #id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        
        # Check confidence
    
        if conf > 0:
            
            print("Inside confidence..........................")
            print("\n")
            
            if conf > 40 and conf <= 85:
                print("Inside face confidence..........................")
                print("\n")
                #id = names[id]
                names = labels[id_]
    
                cv2.putText(img, str(names), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            
            else:
                print("Non-confidence......................")
                make_dataset(img)
                
            
        #else:
            #make_dataset(img)
        
        print("Not inside//////////////////////")
        
        #cv2.putText(img, str(names), (x+5,y-5), font, 1, (255,255,255), 2)
        #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    #k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    #if k == 27:
    #    break


cam.release()
cv2.destroyAllWindows()