import rPPG.csk
import numpy as np
from scipy.misc import imread, imsave
import cv2 # (Optional) OpenCV for drawing bounding boxes
from rPPG.rPPG_preprocessing import *
from rPPG.csk import *
class CSKFaceDetector():
 
    def __init__(self):
        self.face_rect = []
        self.tracker = CSK() # CSK instance
        self.init = True        
    def track_face(self,frame,gray):
        if self.init:
            frame_cropped,gray_cropped,self.face_rect = crop_to_face(frame,gray,[0,0,0,0])
            self.tracker.init(gray,self.face_rect[0],self.face_rect[1],self.face_rect[2],self.face_rect[3])
            self.init = False
            return frame_cropped,gray_cropped,self.face_rect
        else:
            self.face_rect[0], self.face_rect[1] = self.tracker.update(gray) # update CSK tracker and output estimated position
            return crop_frame(frame,self.face_rect),crop_frame(gray,self.face_rect),self.face_rect



# 1st frame's groundtruth information
