"""
Created on Thu Apr  8 13:06:46 2021

@author: limon
"""

import cv2
import os


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('I:/1.232 Pora/ALL Projects/Face Detection/with real training/lmn try all face/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
#face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
p = 0
  
dir_n = "person" + str(p)
os.mkdir(dir_n)
img_count = 1


while(True):

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
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
    elif count >= 20: # Take 30 face sample and stop video
         break
    
    p = p + 1
    
   
     



# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()