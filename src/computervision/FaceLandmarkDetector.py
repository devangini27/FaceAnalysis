'''
Created on 23-Sep-2017

@author: devangini
'''
import dlib
from computervision.FaceDetector import FaceDetector
import numpy as np


class FaceLandmarkDetector:
    """
    This class detects facial landmarks on a face image
    """
    
    def __init__(self):
        #face detector object
        self.faceDetector = FaceDetector()
        
        #path to the landmarks model
        predictor_path = "../models/landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)
        
    
    def findFacialLandmarks(self, image):
        """
        This method finds the facial landmarks
        """
        
        #find the face
        faceBox = self.faceDetector.detectFace(image)
        
        if faceBox != None:
            # Get the landmarks/parts for the face in box d.
            #find the landmarks
        
            points = self.predictor(image, faceBox)
            landmarks = np.zeros((68, 2), np.int16)
            for index in range(68):
                landmarks[index][0] = points.part(index).x
                landmarks[index][1] = points.part(index).y
            
            
            return landmarks
        return None