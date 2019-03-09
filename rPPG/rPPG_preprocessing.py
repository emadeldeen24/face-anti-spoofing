import numpy as np
import argparse
import cv2
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from rPPG.util.opencv_util import *


lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


def apply_skin_classifier(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    num_skin_pixels = np.sum(np.max(skinMask,1))
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    return skin,num_skin_pixels

def get_subroi_rect(frame_cropped,roi):
    w,h = frame_cropped.shape[:2]
    min_x = int(roi[0] * w)
    max_x = int(roi[1] * w)
    min_y = int(roi[2] * h)
    max_y = int(roi[3] * h)
    return [min_x,min_y,max_x-min_x,max_y-min_y]


def crop_frame(frame,rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    return frame[y:y+h,x:x+w]


def crop_to_face(frame,gray,prev_face):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        if prev_face[0] == 0:
            return frame,gray,prev_face
        else:
            return crop_frame(frame,prev_face),crop_frame(gray,prev_face),prev_face
    else:
        if prev_face[0] == 0:
            face_rect = faces[0]
            return crop_frame(frame,faces[0]),crop_frame(gray,faces[0]),faces[0]
        else:
            face_rect = faces[0]

            delta = (face_rect[0] - prev_face[0])**2 + (face_rect[1] - prev_face[1])**2 
            if delta > 40:
                face_rect[2] = prev_face[2]
                face_rect[3] = prev_face[3]
                return crop_frame(frame,face_rect),crop_frame(gray,face_rect),face_rect
            else:
                return crop_frame(frame,prev_face),crop_frame(gray,prev_face),prev_face

#def crop_to_face_PI(frame,gray,prev_face,e_sum):
#    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    kp = 1
#    ki = 0
#    if len(faces) == 0:
#        if len(prev_face) == 0:
#            return frame,gray,None
#        else:
#            return crop_frame(frame,prev_face),crop_frame(gray,prev_face),prev_face
#    else:
#        if len(prev_face) == 0:
#            return crop_frame(frame,faces[0]),crop_frame(gray,faces[0]),faces[0]
#        else:
#            face_rect = faces[0]
#            ey = face_rect[0] - prev_face[0]
#            ex = face_rect[1] - prev_face[1]

#            e_sum = [e_sum[0] + ex, e_sum[1] + ey]
#            print(e_sum)

#            face_rect[0] = prev_face[0] + e_sum[0] * ki + ey * kp
#            face_rect[1] = prev_face[1] + e_sum[1] * ki + ex * kp
                        
#            face_rect[2] = prev_face[2]
#            face_rect[3] = prev_face[3]
#            return crop_frame(frame,face_rect),crop_frame(gray,face_rect),face_rect

