'''
Created on 06-May-2017

@author: devangini
'''

import math
import numpy as np


class ProcrustesAnalysis:
    """
    This class performs Procrustes Analysis on facial landmarks
    """
   
    def __init__(self):
        #stationary landmarks, i.e. left ear, chin, right ear
        self.stationaryLandmarks = [0, 16, 27]
        
        self.debug = True
        
     
    def findScale(self, points, mean):
        """
        This method finds the distance of all the points from the mean
        """
        
        scale = 0
        for point in points:
            scale += (point[0] - mean[0]) ** 2 + (point[1] - mean[1]) ** 2
        scale = math.sqrt(scale / len(points))
        return scale
    
    
    def findFaceOrientation(self, landmarks):
        """
        This method finds the orientation of the face
        """
        #find the slope of the line connecting the two outer eye drops
        index1 = 36
        index2 = 45
        dy = landmarks[index2][1] - landmarks[index1][1]
        dx = landmarks[index2][0] - landmarks[index1][0]
        angle = math.atan2(dy, dx)
        return angle
        
    
    def findRotationAngle(self, landmarks2, lastLandmarks):
        """
        This finds the angle of difference between two pairs of landmarks
        """
        #find the orientation of both landmarks
        angle1 = self.findFaceOrientation(lastLandmarks)
        angle2 = self.findFaceOrientation(landmarks2)
        
        #angle of difference is the difference of the two angles
        theta = angle1 - angle2
        return theta
        
    
    def removeHeadMotion(self, landmarks, lastLandmarks):
        """
        This method removes the effect of head motion from motion vectors
        Uses Procrustes Analysis
        """
        landmarks2 = landmarks.copy()
        
        #these landmarks don't move
        points1 = lastLandmarks[self.stationaryLandmarks, :]
        points2 = landmarks[self.stationaryLandmarks, :]
        
        #find their means
        mean1 = np.mean(points1, axis = 0)
        mean2 = np.mean(points2, axis = 0)
        
        #find the translation
        translation = mean1 - mean2
        
        # translate landmarks
        for index in range(len(landmarks)):
            landmarks2[index, :] += translation
        for index in range(len(points2)):
            points2[index, :] += translation
        
        #find the scale of both faces
        scale1 = self.findScale(points1, mean1)
        scale2 = self.findScale(points2, mean1)
        
        #find the scale
        scale = scale1 / scale2
        
        #scale landmarks to same scale
        for index in range(len(landmarks)):
            landmarks2[index, :] = mean1[:] + (landmarks2[index, :] - mean1) * scale
        for index in range(len(points2)):
            points2[index, :] = mean1[:] + (points2[index, :] - mean1) * scale
       
        #find rotation
        theta = self.findRotationAngle(landmarks2, lastLandmarks)
        
        #rotate landmarks
        for index in range(len(landmarks)):
            w = (landmarks2[index, 0] - mean1[0])
            z = (landmarks2[index, 1] - mean1[1])
            landmarks2[index, 0] = mean1[0] + (w * math.cos(theta) - z * math.sin(theta))
            landmarks2[index, 1] = mean1[1] + (w * math.sin(theta) + z * math.cos(theta))
  
        
        return landmarks2
        
        