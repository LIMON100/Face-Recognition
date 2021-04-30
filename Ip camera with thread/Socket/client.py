# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:22:00 2021

@author: limon
"""

#from camera import CameraStream
import socket,cv2, pickle,struct
import pyshine as ps # pip install pyshine
import imutils # pip install imutils
camera = True

#rtsp://admin:Experts@202@24.186.96.191:554/ch01/0
#"http://158.58.130.148:80/mjpg/video.mjpg"
#vid = cv2.VideoCapture("rtsp://admin:Experts@!@@24.186.96.191:554/ch01/0")


## Check with another camera stream

#vid = CameraStream().start()


def fnu_rtsp():

    user_name = input('Enter Username: ')
    passwrd = input('Enter Password: ')
    ip_address = input('Enter IP_Address: ')
    port = input('Enter Port: ')
    channel = input('Enter Channel: ')
    
    new_url = "rtsp://" + user_name + ":" + passwrd + "@" + ip_address + ":" + port + "/" + "ch01" + "/" + channel
    #print(new_url)
    
    video = cv2.VideoCapture(new_url)
    return video


vid = fnu_rtsp()


client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = 'local_ip' 

port = 9999
client_socket.connect((host_ip, port))

if client_socket: 
    
	while (vid.isOpened()):
		try:
			img, frame = vid.read()
			frame = imutils.resize(frame,width=380)
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			cv2.imshow(f"TO: {host_ip}",frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				client_socket.close()
                
		except:
			print('VIDEO FINISHED!')
			break