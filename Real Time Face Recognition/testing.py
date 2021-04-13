# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:15:53 2021

@author: limon
"""


import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set video widht
cap.set(4, 480)

minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.5, 
        minNeighbors = 5,
        minSize = (int(minW), int(minH)))
    
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
        #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=4 and conf <= 85:
    		#print(5: #id_)
    		#print(labels[id_])
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, str(name), (x-10, y-10), font, 1, (255,255,255), 2)
            #cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)

    	#img_item = "7.png"
    	#cv2.imwrite(img_item, roi_color)

    	#color = (255, 0, 0) #BGR 0-255 
    	#stroke = 2
    	#end_cord_x = x + w
    	#end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    	#subitems = smile_cascade.detectMultiScale(roi_gray)
    	#for (ex,ey,ew,eh) in subitems:
    	#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()