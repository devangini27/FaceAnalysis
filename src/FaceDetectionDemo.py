'''
Created on 23-Sep-2017

@author: devangini
'''
from computervision.FaceDetector import FaceDetector
import cv2

#This is a demo for face detection

if __name__ == "__main__":
    faceDetector = FaceDetector()
    
    video = cv2.VideoCapture(0)
    while True:
        ret, image = video.read()
        image = cv2.flip(image,1)
        
        if image == None:
            break
        
        faceBox = faceDetector.detectFace(image)
        print faceBox
        if faceBox != None:
            cv2.rectangle(image,(faceBox.left(),faceBox.top()),(faceBox.right(),faceBox.bottom()),(0,255,0),3)

        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break