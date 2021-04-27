# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:37:46 2021

@author: limon
"""


from threading import Thread
import time
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import random

face_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/cascades/data//haarcascade_eye.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
global p
p = 0
save_value = 0


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
    #global faces
    
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
                
                print("Collect Faces............")
                faces = face_cascade.detectMultiScale(image_array)
                
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    
    
    print("Saving class names.......................")
    print("\n")
    with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/pickles/face-labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)
    
    
    print("Save file to yml..........................")
    print("\n")
    recognizer.train(x_train, np.array(y_labels))
    recognizer.write("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/recognizers/face-trainner.yml")
    
    print("Finish Training--------------------------")
    print("\n")




def make_dataset(img):

    print("Make New Dataset.......................")
    
    #p = q
    count = 0
    #global p
    #global save_value
    
    #a.append(p)
    #b = a[-1] + 12
    #b = p
    #print(b)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, 'new_dataset')
    
    for path, dirs,files in os.walk(image_dir):
        a1 = os.path.basename(path)
     
        
    b1 = a1[-1]
    b1 = (int(b1) + 1) * 2 + random.randint(0, 5555555)
    b1 = str(b1)
    
    dir_n = "I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/new_dataset/p" + str(b1)
    os.mkdir(dir_n)
    img_count = 1
    
    with open("ltest.txt", "r") as fp:
        lines = fp.readlines()
        end = lines[-1].split(',')[0]
    
    save_value = end
    
    while(True):
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            
            #os.mkdir(pathn)
            #pathn = os.makedirs(os.path.join('subfolder' + str(p)))
            #cv2.imwrite(pathn + str(face_id) + '.' + str(count) + ".jpg", rgb[y:y+h,x:x+w])
            cv2.imwrite(dir_n + f"\image{img_count}.jpg", rgb[y:y+h,x:x+w])
    
            #cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        img_count += 1
        #p = p + 1
        
        if k == 27:
            break
        elif count >= 30:
            #p = p + 1
            #a.append(p)
            break
    
    
    print("Complete Making Dataset.....................")
    print("Goes to Training")
    print("\n")
    train_data()


class VideoStreamWidget(object):
    
    def __init__(self, src = 0, width = 640, height = 480):
        
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.thread = Thread(target=self.update, args=())
        self.minW = 0.1*self.capture.get(3)
        self.minH = 0.1*self.capture.get(4)
        self.thread.daemon = True
        self.thread.start()
    
    
        recognizer.read("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/recognizers/face-trainner.yml")
        #labels = {"person_name": 1}
        with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/pickles/face-labels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                self.labels = {v:k for k,v in og_labels.items()}


    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                
                self.gray  = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.faces = face_cascade.detectMultiScale(
                    self.gray, 
                    scaleFactor=1.5, 
                    minNeighbors = 5,
                    minSize = (220, 220))
                
                
                for(x, y, w, h) in self.faces:
        
                    #flag = 0
                    cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
                    #id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                    
                    self.id_, self.conf = recognizer.predict(self.gray[y:y+h, x:x+w])
                    
                    
                    # Check confidence
                
                    if self.conf > 0:
                        
                        print("Inside confidence..........................")
                        print("\n")
                        
                        if self.conf > 40 and self.conf <= 85:
                            print("Inside face confidence..........................")
                            print("\n")
                            #id = names[id]
                            self.names = self.labels[self.id_]
                
                            cv2.putText(self.frame, str(self.names), (x+5,y-5), font, 1, (255,255,255), 2)
                            #cv2.putText(img, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)  
                        
                        else:
                            print("Non-confidence......................")
                            make_dataset(self.frame)
                            
                        
                    #else:
                        #make_dataset(img)
                    
                    print("Not inside//////////////////////")
                
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(0)
            

if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass