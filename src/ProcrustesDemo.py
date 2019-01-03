'''
Created on 23-Sep-2017

@author: devangini
'''
import cv2
from src.computervision.ProcrustesAnalysis import ProcrustesAnalysis
from src.computervision.FaceLandmarkDetector import FaceLandmarkDetector

#This is a demo for face detection

if __name__ == "__main__":
    faceLandmarkDetector = FaceLandmarkDetector()
    procrustes = ProcrustesAnalysis()
    
    video = cv2.VideoCapture(0)
    lastLandmarks = None
    
    while True:
        ret, image = video.read()
        image = cv2.flip(image,1)
        
        if image == None:
            break
        
        #find landmarks
        landmarks = faceLandmarkDetector.findFacialLandmarks(image)
        
        if lastLandmarks != None:
            
            if landmarks != None:
            
                #procrustes analysis
                landmarks2 = procrustes.removeHeadMotion(landmarks, lastLandmarks)
        else:
            if landmarks != None:
                lastLandmarks = landmarks.copy()
                

        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break