'''
Created on 06-May-2017

@author: devangini
'''
from computervision.FacialExpressions import FacialExpressions
import cv2


if __name__ == "__main__": 
    #video = cv2.VideoCapture('../output.avi')
    
    video = cv2.VideoCapture(0)
    
    
    facialExpr = FacialExpressions()
    facialExpr.findFacialExpressions(video, True)