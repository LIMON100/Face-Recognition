# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:13:08 2021

@author: limon
"""


from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
	args["server_ip"]))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
rpiName = socket.gethostname()
# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


while True:
    frame = vs.read()
    #cv2.imshow("Client", frame)
    sender.send_image(rpiName, frame)