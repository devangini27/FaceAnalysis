'''
Created on 23-Sep-2017

@author: devangini
'''
import dlib


class FaceDetector:
    """
    This class detects a face in the image
    """
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        
    
    def detectFace(self, image):
        """
        This method detects a face in the image
        It returns the first rectangle ROI containing a face
        """
        dets = self.detector(image, 0)
        for _, detectedBox in enumerate(dets):
            return detectedBox
        return None