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
#import trainImg
#import train2
import makeData
#import makeData2
import argparse
#import imagezmq
#import socket
import pika




face_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/cascades/data//haarcascade_eye.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

    
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'new_dataset')


class VideoStreamWidget(object):
    
    def __init__(self, src = 0, width = 640, height = 480, queue='hello', host='localhost', routingKey='hello', exchange=''):
        
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.queue = queue
        self.host = host
        self.routingKey = routingKey
        self.exchange = exchange
        
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host = self.host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue = self.queue)
        
        #self.channel.basic_publish(exchange=self.exchange, routing_key=self.routingKey, body="SENT ITEM")

        #print("Published Message:")
        #self.connection.close()
        
        self.thread = Thread(target=self.update, args=())
        self.minW = 0.1*self.capture.get(3)
        self.minH = 0.1*self.capture.get(4)
        self.thread.daemon = True
        self.thread.start()
        
        recognizer.read("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/recognizers/face-trainner.yml")
        labels = {"person_name": 1}
        with open("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/pickles/face-labels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {v:k for k,v in og_labels.items()}
        
    
    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                
                self.gray  = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.faces = face_cascade.detectMultiScale(
                    self.gray, 
                    scaleFactor=1.5, 
                    minNeighbors = 5,
                    minSize = (64, 48))
                    #minsize = (int(self.minW), int(self.minH)))
                    #minSize = (self.min, 220))
                
                for(x, y, w, h) in self.faces:
        
                  
                    cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    
                    self.id_, self.conf = recognizer.predict(self.gray[y:y+h, x:x+w])
                    
                    
                    # Check confidence
                
                    if self.conf > 0:
                        
                        print("Inside confidence..........................")
                        print("\n")
                        
                        if self.conf > 40 and self.conf <= 85:
                            print("Inside face confidence..........................")
                            print("\n")
                            #id = names[id]
                            self.names = labels[self.id_]
                            #sender.send_image(rpiName, self.frame)
                            #cv2.putText(str(self.names), (x+5,y-5), font, 1, (255,255,255), 2)
                            cv2.putText(self.frame, str(self.names), (x+5,y-5), font, 1, (255,255,255), 2)
                            #cv2.putText(img, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)  
                        
                        else:
                            print("Non-confidence......................")
                            #cv2.rectangle(self.frame, (x,y), (x+w,y+h), (128,0,0), 2)
                            #md = makeData.MakeDataset(self.frame)
                            #md.update_data()
                            
                            
                            count = 0
                            img_count = 1
                           
                   
                            result = []
                            
                            #mypath = "I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/cvs/video_dataset"
                    
                            for root, dirs,files in os.walk(image_dir):
                                for file in files:
                                    path = os.path.join(root, file)
                                    a1 = os.path.basename(root).replace(" ", "-").lower()
                                    b = int(re.search(r'\d+', a1).group())
                                    result.append(b)
                                
                    
                         
                            b1 = max(result) + 1
                            
                            dir_n = "I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/RabbitMq/with2/new_dataset/p" + str(b1)
                            os.mkdir(dir_n)
                            
                            #c1 = str(self.b1)
                            #self.cam = cv2.VideoCapture("I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/All in/cvs/video_dataset" + f"\p{self.c1}.mp4")
                        
                            print("Make New Dataset.......................")
                            
                            while(True):
                                count += 1
                                            
                                cv2.imwrite(dir_n + f"\image{img_count}.jpg", self.rgb[y:y+h,x:x+w])
                                    
                                #cv2.imshow('image', self.frame)
                                    
                                k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
                                img_count += 1
                                
                                if k == 27:
                                    break
                                elif count >= 3:
                                    break
                            
                            self.channel.basic_publish(exchange = self.exchange, routing_key = self.routingKey, body="SENT ITEM")

                            print("Published Message:")
                            #self.connection.close()
                            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
                            #channel = connection.channel()
                            
                            #channel.queue_declare(queue='hello')
                            
                            #channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')
                            #print("[x] Sent 'Hello World!'")
                            #connection.close()
                            #sender.send_image(rpiName, self.frame)
                            #td = train2.TraningData()
                            #td.train_data()
                            
                   
                        #td = train2.TraningData()
                        #td.train_data()
                            #md.train_data()
                            
                print("Not inside//////////////////////")
                
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            self.connection.close()
            exit(0)
            


if __name__ == '__main__':
    
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass