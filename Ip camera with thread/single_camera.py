# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:11:48 2021

@author: limon
"""


import cv2
import threading, time
import queue

input_buffer = queue.Queue()

def processing():
    
    while True:
        print("get")
        frame = input_buffer.get()
        cv2.imshow("Video", frame)
        time.sleep(0.025)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


def fnu_rtsp():

    user_name = input('Enter Username: ')
    passwrd = input('Enter Password: ')
    ip_address = input('Enter IP_Address: ')
    port = input('Enter Port: ')
    channel = input('Enter Channel: ')
    
    new_url = "rtsp://" + user_name + ":" + passwrd + "@" + ip_address + ":" + port + "/" + "/" + "ch01" + "/" + channel
    
    video = cv2.VideoCapture(new_url)
    return video


rs = fnu_rtsp()

t = threading.Thread(target = processing)
t.daemon = True
t.start()

while True:
    ret, frame = rs.read()
    input_buffer.put(frame)
    time.sleep(0.025)
    print("put")