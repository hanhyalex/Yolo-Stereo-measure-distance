
import cv2
import numpy as np
import os
import sys
from PIL import Image
import sys
import argparse
from yolo import YOLO, detect_video 
from PIL import Image
import cv2 as cv
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from PIL import Image, ImageFont, ImageDraw
from PyQt5.QtGui import QPixmap, QImage

i = 1
# cap = cv2.VideoCapture(i)
cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
yolo=YOLO()
yolo.detect_my_video(yolo,cap)
# while cap.isOpened():
#     ret, frame = cap.read()
#     size=[480,640,3]
#     size_2=[640,480,3]
#     left_img = cv2.resize(frame[:, 0:int(size[1]/2), :],(640,480))
#     right_img = cv2.resize(frame[:, int(size[1]/2):size[1], :],(640,480))
# #    left_img = left_img[:,:,0:3]
#     if ret:
#         # 显示两幅图片合成的图片
#         #cv2.imshow('img', frame)
#         # 显示左摄像头视图
#         cv2.imshow('left',left_img)
#         # 显示右摄像头视图
#         cv2.imshow('right', right_img)
#         # cv2.imshow('frame', frame)
# #        edges = cv2.Canny(left_img,100,200)
# #        cv2.show()
# #        cv2.waitKey(0)
        
        
#     key = cv2.waitKey(delay=2)
#     if key == ord("t"):
#         # cv2.imwrite('/home/pi/Desktop/Python/left/' + str(i) + '.jpg', left_img)#
#         # cv2.imwrite('/home/pi/Desktop/Python/right/' + str(i) + '.jpg', right_img)#
#         print('yes')
#         i += 1
#     if key == ord("q") or key == 27:
#         break
#     # if key == ord("s"):
#     #     print('start recording')
#     #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     #     print(size)
#     #     # out_l = cv2.VideoWriter('camera_left.avi', fourcc,20.0, size)
#     #     # out_r = cv2.VideoWriter('camera_right.avi', fourcc,20.0, size)
#     #     while True :
#     #             ret, frame = cap.read()
#     #             left_img = frame[:, 0:size[0], :]
#     #             right_img = frame[:, size[0]:size[0]*2, :]
#     #             if ret:
#     #             # 显示两幅图片合成的图片
#     #             #cv2.imshow('img', frame)
#     #             # 显示左摄像头视图
#     #                 cv2.imshow('left', left_img)
#     #                 # 显示右摄像头视图
#     #                 cv2.imshow('right', right_img)
#     #             out_l.write(left_img)
#     #             out_r.write(right_img)
#     #             key = cv2.waitKey(delay=2)
#     #             if key == ord("q") or key == 27:
#     #                 print('stop recording')
#     #                 out_l.release()
#     #                 out_r.release()
#     #                 break



# #frame.release()
# cap.release()

# cv2.destroyAllWindows()