'''
Created on 23-Sep-2017

@author: devangini
'''
import cv2
from computervision.FaceLandmarkDetector import FaceLandmarkDetector

#This is a demo for facial landmarks detection

if __name__ == "__main__":
    faceLandmarkDetector = FaceLandmarkDetector()
    
    video = cv2.VideoCapture(0)
    while True:
        ret, image = video.read()
        image = cv2.flip(image,1)
        
        if image == None:
            break
        
        landmarks = faceLandmarkDetector.findFacialLandmarks(image)

        if landmarks != None:
            for i in range(68):
                cv2.circle(image,(landmarks[i,0],landmarks[i,1]), 3, (0,0,255), -1)

        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break