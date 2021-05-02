# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:08:52 2021

@author: limon
""" 

import time
import numpy as np
import cv2
import pickle
import os
from PIL import Image
import random
import re
from threading import Thread
import imagezmq

face_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/cascades/data//haarcascade_eye.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/recognizers/face-trainner.yml")

    
font = cv2.FONT_HERSHEY_SIMPLEX

current_id = 0
label_ids = {}
y_labels = []
x_train = []
a = []

imageHub = imagezmq.ImageHub()


class TraningData:
    
    #def __init__(self):
    #    (self.rpiName, self.frame) = imageHub.recv_image()
    
    def train_data(self):
        
        print("Start Training...............................")
        
        (self.rpiName, self.frame) = imageHub.recv_image()
        
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
                    
                    #print("Collect Faces............")
                    faces = face_cascade.detectMultiScale(image_array)
                    
                    for (x, y, w, h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)
                        
        
        print("Saving class names.......................")
        print("\n")
        with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/pickles/face-labels.pickle", "wb") as f:
            pickle.dump(label_ids, f)
        
        
        print("Save file to yml..........................")
        print("\n")
        recognizer.train(x_train, np.array(y_labels))
        recognizer.write("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/recognizers/face-trainner.yml")
        
        print("Finish Training--------------------------")
        print("\n")
    


