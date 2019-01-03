'''
Created on 05-May-2017

@author: devangini
'''
import cv2
import dlib
import numpy as np
from computervision.ProcrustesAnalysis import ProcrustesAnalysis
from sklearn.externals import joblib
import math
import datetime


class FacialExpressions:
    
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "../models/landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)
        
        self.lastLandmarks = None
        
        self.procustesAnalysis = ProcrustesAnalysis()
        
       
        self.AUGroups = [[19, 20, 21, 22, 23, 24], # AU 1
                    [17, 18, 19, 24, 25, 26], # AU 2
                    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26], # AU 4
                    [31, 32, 33, 34, 35], # AU 9
                    [48, 54], #AU 12
                    [48, 54], #AU 15
                    [65, 66, 67] # AU 26
                    ] 
        self.AUDirection = [[0, -1], # AU 1 
                            [0, -1], # AU 2 
                            [0, 1], # AU 4 
                            [0, -1], # AU 9 
                            [0, -1], #AU 12 
                            [0, 1], #AU 15 
                            [0, 1]# AU 26 
                            ]
        self.FEGroups = [[4], #happiness
                         [0, 2, 5], #sadness
                         [0, 1, 3, 6] # surprise
                         ] 
      
        self.ipdDistance = 0
 
        self.FEModel = joblib.load('../models/FEModel5.pkl')
        
    
    def findFacialLandmarks(self, image):
        
        #find the face

        dets = self.detector(image, 0)
        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            #find the landmarks
        
            points = self.predictor(image, d)
            landmarks = np.zeros((68, 2), np.float32)
            for index in range(68):
                landmarks[index][0] = points.part(index).x
                landmarks[index][1] = points.part(index).y
            
            
            return landmarks
        return None
    
    
    
    def findIpdDistance(self):
        #find ipd of first landmarks
        #distance between 36 and 45
        distx = (self.lastLandmarks[45][0] - self.lastLandmarks[36][0])
        disty = (self.lastLandmarks[45][1] - self.lastLandmarks[36][1])
        distance = math.sqrt(distx ** 2 + disty ** 2)
        self.ipdDistance = distance
        

    def groupAUs(self, motionVectors):
        """
        This function groups the motion vectors based on AUs
        """
        auVectors = np.zeros((len(self.AUGroups), 2), dtype = np.float16)
        #sum of all motion vectors in different au groups
        index = 0
        for group in self.AUGroups:
            total = np.zeros((1, 2), dtype = np.float16)
            for index2 in group:
                total += motionVectors[index2, :]
            auVectors[index, : ] = total / len(group)
            index += 1
        return auVectors
        

    def computeAUIntensity(self, auVectors):
        """
        This function computes the intensity of AUs
        """
        AUIntensities = []
        
        for index in range(len(self.AUGroups)):
            #perform dot product of AU vector with direction vector of AU
            dotProduct = auVectors[index][0] * self.AUDirection[index][0] + auVectors[index][1] * self.AUDirection[index][1]
            dotProduct = max(0, dotProduct)
            
            AUIntensities.append(dotProduct)
            
        return AUIntensities
    
        
    def computeFEIntensity(self, AUIntensities):
        """
        This function computes the facial expression intensity
        """
        FEIntensities = []
        for feindexes in self.FEGroups:
            intensity = 0
            for auindex in feindexes:
                intensity += AUIntensities[auindex]
            FEIntensities.append(intensity / len(feindexes))
        return FEIntensities
    
    
        
    def classifyFE(self, FEIntensities):
        """
        This function classes FE
        """
        FEIntensities = np.asarray(FEIntensities)
        FEIntensities = FEIntensities.reshape((1, -1))
        predLabels = self.FEModel.predict(FEIntensities)
  
        #return predLabels
    
        predLabels = 0
        # View probabilities=
        probas = self.FEModel.predict_proba(FEIntensities)
        
        probability = 0
        if probas[0][0] > 0.5:
            predLabels = 0
            probability = probas[0][0]
        elif probas[0][2] > 0.4:
            predLabels = 2
            probability = probas[0][2]
        elif probas[0][1] > 0.60:
            predLabels = 1
            probability = probas[0][1]
        else:
            predLabels = 3
            probability = 1 - probas[0][1]
        
        return predLabels, probability


    
    def visualizeEmotions(self, image, emoLabel, landmarks):
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #DRAW LANDMARKS
        for index in range(68):
            cv2.circle(image,(int(landmarks[index][ 0]), int(landmarks[index][ 1])), 2, (0,255,0), -1)
            #cv2.circle(image,(int(self.lastLandmarks[index][ 0]), int(self.lastLandmarks[index][ 1])), 2, (0,0,255), -1)
            #cv2.circle(image,(int(landmarks2[index][ 0]), int(landmarks2[index][ 1])), 2, (255,0,0), -1)
    
                
         
        
        if emoLabel == 0:
            cv2.circle(image,(100, 50), 20, (0,255,0), -1)
            cv2.putText(image,'happy',(130,50), font, 2,(0,255,0),2,cv2.LINE_AA)
        elif emoLabel == 1:
            cv2.circle(image,(100, 50), 20, (0,0,255), -1)
            cv2.putText(image,'sad',(130,50), font, 2,(0,0,255),2,cv2.LINE_AA)
        elif emoLabel == 2:
            cv2.circle(image,(100, 50), 20, (255,0,0), -1)
            cv2.putText(image,'surprise',(130,50), font, 2,(255,0,0),2,cv2.LINE_AA)
        else:
            cv2.circle(image,(100, 50), 20, (0,0,0), -1)
            cv2.putText(image,'neutral',(130,50), font, 2,(0,0,0),2,cv2.LINE_AA)
            
         
    def findFacialExpression(self, image):
        
        
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #find landmarks
        landmarks = self.findFacialLandmarks(imageGray)
        
        if self.lastLandmarks is not None:
            
            if landmarks is not None:
            
                #procrustes analysis
                landmarks2 = self.procustesAnalysis.removeHeadMotion(landmarks, self.lastLandmarks)
                
                
                #find motion vectors
                #find the difference in landmarks
                motionVectors = landmarks2 - self.lastLandmarks
            
                #groups aus
                auVectors = self.groupAUs(motionVectors)
                AUIntensities = self.computeAUIntensity(auVectors)
                
                #group aus to find fe
                FEIntensities = self.computeFEIntensity(AUIntensities)
                
               
                #divide intensity by ipd distance to normalize fe intensity
                #find ipd distance
                for index in range(len(FEIntensities)):
                    FEIntensities[index] /= self.ipdDistance
           
                
                
                
                #find probability of fe
                #perform svm test
                emoLabel, probability  = self.classifyFE(FEIntensities)
                
                #visualize
                self.visualizeEmotions(image, emoLabel, landmarks)
                
                #return FEIntensities
                return emoLabel, probability, landmarks
            
            return None, None, None
        else:
            if landmarks is not None:
                self.lastLandmarks = landmarks.copy()
                self.findIpdDistance()
       
            return None, None, None
    
    def findFacialExpressions(self, video, show):
        frameCount = 0
        labels = []
        probs = []
        
       
        start = datetime.datetime.now()
        
        while(video.isOpened()):
            ret, frame = video.read()
            frame = cv2.flip(frame,1)
            
            if frame is None:
                break
            

            
            emoLabel, probability, _ = self.findFacialExpression(frame)
            labels.append(emoLabel)
            probs.append(probability)
            
            frameCount += 1
            
            if show:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        end = datetime.datetime.now()
            
        print "frameCount", frameCount 
        elapsed = (end - start).total_seconds()
        print "fps", frameCount / elapsed

        return labels, probs
    

if __name__ == "__main__": 
    video = cv2.VideoCapture('../output.avi')
    
    #video = cv2.VideoCapture(0)
    
    facialExpr = FacialExpressions()
    facialExpr.findFacialExpressions(video)
    
    