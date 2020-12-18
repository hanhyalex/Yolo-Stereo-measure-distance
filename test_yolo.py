# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:28:45 2020

@author: 86198
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:05:49 2020

@author: 86198
"""
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

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''

parser.add_argument(
    '--model', type=str,
    help='path to model weight file, default ' + YOLO.get_defaults("model_path")
)

parser.add_argument(
    '--classes', type=str,
    help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
)

parser.add_argument(
    '--gpu_num', type=int,
    help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
)

parser.add_argument(
    '--image', default=False, action="store_true",
    help='Image detection mode, will ignore all positional arguments'
)
'''
Command line positional arguments -- for video detection mode
'''

parser.add_argument(
    "--output", nargs='?', type=str, default="",
    help = "[Optional] Video output path"
)
FLAGS = parser.parse_args()
yolo=YOLO(**vars(FLAGS))


#camera_configs
left_camera_matrix = np.array([[725.418207743281,	0,	0],
                                         [0.566232546068237,	725.144199850458,	0],
                                         [605.134233318879,	352.856123759109,	1]]).T
left_distortion = np.array([[0.0323753321944668,	-0.0742753878448274,	0.000527201631996113,	0.000468360715393419,	0.0644021561263970]])


right_camera_matrix = np.array([[ 723.704166672267,	0,	0],
                                          [1.08041552239949,	723.026835518584,	0],
                                          [602.303048868564,	354.402934083807,	1]]).T
right_distortion = np.array([[0.0279896446080107,	-0.107203500123106, 0.000992573004150838,	0.00113416150708816,0.135870309164238]])
 
# 旋转关系向量

R = np.array([[0.999994879074946,	0.00306010500280413,	-0.000936793069934069],
                           [-0.00305593583853680,	0.999985562289845,	0.00442000769383407],
                           [0.000950305232443632,	-0.00441712227979022,	0.999989792923273]]).T  # 使用Rodrigues变换将om变换为R
T = np.array([[-119.765127208902], [0.237341173315274], [-1.09956215996867	]])# 平移关系向量

 


size = (1280, 720) # 图像尺寸
 
# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T,alpha=0)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

l_img=r'C:\Users\86198\Desktop\26.jpg'
r_img=r'C:\Users\86198\Desktop\26r.jpg'
# l_img=r'C:\Users\86198\Desktop\0.jpg'

frame1 = cv2.imread(l_img)   # 左图
frame2 = cv2.imread(r_img)   # 右图
 
 
# 根据更正map对图片进行重构
img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
img1_rectified=cv2.cvtColor(img1_rectified,cv2.COLOR_BGR2RGB)
# 将图片置为灰度图，为StereoBM作准备
imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

# SGBM
window_size = 7
# min_disp = 0
# num_disp = 320 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=192,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=7,
    P1=8 * 3 * window_size ** 2,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=16 * 3 * window_size ** 2,
    disp12MaxDiff=3,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(imgL, imgR)
#将深度图在进行3D转换前操作的两种方式  
#    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
disp = disparity.astype(np.float32) / 16.0
# 将图片扩展至3d空间中，其z方向的值则为当前的距离
threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)
not_neg =  threeD [:,:,2]>0 
not_neg=np.array(not_neg, dtype=np.float32)
threeD=threeD[:,:,2]*not_neg
threeD[threeD==np.inf] = 0
shape = threeD.shape
final_threeD = np.zeros((shape))



# img = r"C:\Users\86198\Desktop\learn\AAAAAAAA相机程序\毕业所需\keras-yolo3-master\32.jpg"
try:
    # image = Image.open(l_img)
    image=Image.fromarray(img1_rectified)
    img_ori=img1_rectified
    Img_ori=Image.fromarray(img1_rectified)
except:
    print('Open Error! Try again!')
else:
    r_image= yolo.detect_image(image)
    image=r_image[0]
    all_boxes=r_image[1]
    print(all_boxes)
    # r_image.show()
    plt.figure(figsize=(50,50))
    plt.imshow(image)
# ----------rgb算法---------------

img_pr = 2.2 *img_ori[:,:,1]-img_ori[:,:,0]-1.5*img_ori[:,:,2]
threeD=threeD*(img_pr>0)
disp=disp*(img_pr>0)
img_ori[:,:,0]=img_ori[:,:,0]*(img_pr>0)
img_ori[:,:,1]=img_ori[:,:,1]*(img_pr>0)
img_ori[:,:,2]=img_ori[:,:,2]*(img_pr>0)
# -------------------------------
cv2.imwrite(r"C:\Users\86198\Desktop\temp.jpg",threeD)
for box in all_boxes:
    box_shape = {'leftstart' : [box[1],box[0]] , 'hight' :int(box[3]-box[1]) , 'width' :int(box[2]-box[0])}
    img1= threeD
    for i in range(box_shape['leftstart'][0],box_shape['leftstart'][0]+box_shape['hight']):
        for j in range(box_shape['leftstart'][1],box_shape['leftstart'][1]+box_shape['width']):
            final_threeD[i,j]=disp[i,j]
    
    img_tar = img1[box_shape['leftstart'][0]: box_shape['leftstart'][0]+box_shape['hight'],box_shape['leftstart'][1]: box_shape['leftstart'][1]+box_shape['width']]
    img_tar2 = final_threeD[box_shape['leftstart'][0]: box_shape['leftstart'][0]+box_shape['hight'],box_shape['leftstart'][1]: box_shape['leftstart'][1]+box_shape['width']]
    img_tar3= img_ori[box_shape['leftstart'][0]: box_shape['leftstart'][0]+box_shape['hight'],box_shape['leftstart'][1]: box_shape['leftstart'][1]+box_shape['width']]
    
    
    # img_tar2=cv.resize(img_tar2,(box_shape['width'],box_shape['hight']))
    # plt.figure(figsize=(5,5))
    # plt.imshow(img_tar)
    
    # plt.figure(figsize=(5,5))
    # plt.imshow(img_tar)
    # -----------------------聚类部分----------------------------------
    # m=0.8
    # # box_shape = {'leftstart' : [325,850] , 'hight' :(int)(box_shape['hight']*m) , 'width' :(int)(box_shape['width']*m)}
    # box_shape = {'leftstart' : box_shape['leftstart'] , 'hight': (int)(box_shape['hight']*m), 'width' :(int)(box_shape['width']*m)}
    
    # img_tar=cv.resize(img_tar,(box_shape['width'],box_shape['hight']))
    # X=[]
    # for i in range(0,box_shape['width']):
    #     for j in range(0,box_shape['hight']):
    #         X.append(np.array([i/box_shape['width'],j/box_shape['hight'], img_tar[j,i]/img_tar.max()]))
    # time1= time.time()
    # result=DBSCAN(eps=0.04,min_samples=15).fit_predict(X)
    # print('聚类时间为：', time.time()-time1)
    # Y=np.zeros((box_shape['hight'],+box_shape['width']))
    # for i in range (len(X)):
    #     Y[i%box_shape['hight'],(int)(i/box_shape['hight'])]=result[i]
    # final_res=Y==2.0
    # final_res_array=img_tar[final_res]
    # final_res=np.array(final_res, dtype=np.float32)
    # final_res=img_tar*final_res
    # final_res=np.resize(final_res,(len(final_res)))
    # kernel = np.ones((2,2),np.uint8)
    # # 腐蚀
    # final_res= cv2.erode(final_res,kernel)
    # kernel = np.ones((3,3),np.uint8)
    # # 膨胀
    # final_res = cv2.dilate(final_res,kernel)
    # final_res = cv2.dilate(final_res,kernel)
    # plt.figure(figsize=(5,5))
    # plt.imshow(final_res)
    # --------------------------------------------------------
    
    # plt.figure(figsize=(15,15))
    # y, x, _ =plt.hist(np.asarray(img_tar),100,(300,10000))
    plt.figure(figsize=(15,15))
    plt.imshow(img_tar2)
    plt.figure(figsize=(15,15))
    plt.imshow(img_tar3)
    
    counts, bins = np.histogram(np.asarray(img_tar),1000,(300,10000))
    plt.figure(figsize=(15,15))
    plt.hist(bins[:-1], bins, weights=counts)
    
    
    
    # ---------直方图算法---------
    
    counts=np.asarray(counts)
    bins=np.asarray(bins)
    max_2_3 = counts.max()*8/9
    take_into_account=np.argwhere(counts>=max_2_3)

    weight=0
    sum_counts=0
    for index in take_into_account:
        sum_counts+=counts[index]
    
    for index in take_into_account:
        weight+=counts[index]*index
    final_weight=np.nan_to_num(float(weight/sum_counts))
    print(final_weight)
    flo=int(np.floor(final_weight))
    ceil=int(np.ceil(final_weight))
    final_posi=(final_weight-flo)*(bins[ceil]-bins[flo])+bins[flo]
    
    
    
    
    # -----------------------
    
    
    # print(np.argmax(y))
    # print(np.unravel_index(y.argmax(), y.shape))
    
    draw = ImageDraw.Draw(Img_ori)
    # label = '{} {:.2f}'.format("maize", y)
    label=(str(int(final_posi))  +"mm")
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    label_size = draw.textsize(label, font)

    
    top = box_shape['leftstart'][0]
    left = box_shape['leftstart'][1]
    bottom = box_shape['leftstart'][0]+box_shape['hight']
    right =  box_shape['leftstart'][1]+box_shape['width']
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])
    thickness = (Img_ori.size[0] + Img_ori.size[1]) // 300
    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=(255, 0, 0))
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=(255, 0, 0))
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    
    
plt.figure(figsize=(50,50))
plt.imshow(Img_ori)
    # 这里要返回一个最终版，可以绘图的那种



# r_img=r'C:\Users\86198\Desktop\3r.jpg'
# yolo_my_img(l_img,r_img)  
# yolo.close_session()