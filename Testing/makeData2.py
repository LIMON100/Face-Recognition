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
import re
import datetime

face_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/cascades/data//haarcascade_eye.xml')


class FPS:
    def __init__(self):
        # store the start time, end time and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and 
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()



class MakeDataset:
    
    def __init__(self, img):
        self.count = 0
        self.img = img
        #self.b = a[-1] + 12
        #self.dir_n = "I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with class/new_dataset/person" + str(self.b)
        self.img_count = 1
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.BASE_DIR, 'new_dataset')
        self.result = []
        
    
        #self.thread = Thread(target = self.update_data, args=())
        #self.thread.daemon = True
        #self.thread.start()
        
        for root, dirs,files in os.walk(self.image_dir):
            for file in files:
                self.path = os.path.join(root, file)
                self.a1 = os.path.basename(root).replace(" ", "-").lower()
                self.b = int(re.search(r'\d+', self.a1).group())
                self.result.append(self.b)
            

        #self.b1 = self.a1[-1]
        self.b1 = max(self.result) + 1 
        #self.b1 = str(self.b1)
        
        self.dir_n = "I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/with2/new_dataset/p" + str(self.b1)
        os.mkdir(self.dir_n)
            
        #os.mkdir(self.dir_n)
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update_data, args=()).start()
        return self

    
    print("Make New Dataset.......................")
    
    
    def update_data(self):
        
        while(True):
        
            rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(rgb, 1.3, 5)
        
            for (x,y,w,h) in faces:
        
                cv2.rectangle(self.img, (x,y), (x+w,y+h), (0,0,128), 2)     
                self.count += 1
                
                cv2.imwrite(self.dir_n + f"\image{self.img_count}.jpg", rgb[y:y+h,x:x+w])
        
                #cv2.imshow('image', img)
        
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            self.img_count += 1
            #p = p + 1
            
            if k == 27:
                break
            elif self.count >= 25:
                #p = p + 1
                #a.append(p)
                break
            
    def read(self):
        # return the frame most recently read
        return self.frame
    

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    
    print("Complete Making Dataset.....................")
    print("Goes to Training")
    print("\n")
    #train_data()