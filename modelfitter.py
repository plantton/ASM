# Class Align Teeth image
import cv2
import cv2.cv as cv
import numpy as np
import time
import fnmatch
import sys
# Package imutils installed via extra command
import imutils
import numpy.matlib
from PIL import Image
import matplotlib.pyplot as plt
import pandas
import os
from scipy.interpolate import Akima1DInterpolator
from Locator import Locator
from Model import Model

# A class to fit ASM to a given image
# Greyscale profile along normal direction of landmarks
# Use mahalanobis distance to calculate similarities

class ModelFitter:
    
    
    def __init__(self, loc,image):
        # Loc is the initial location of Active Shape Model
        # Loc can be given by class Locator (automatic initialisation)
        # Loc can also be given by manual initialization (mouse input)
        self.loc = loc
        file_in = image
        self.img = cv2.imread(file_in)
        
    def __produce_gradient_image(self):
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(25,25))
        cl1=clahe.apply(gray)
        blur = cv2.GaussianBlur(cl1,(3,3),0)
        blur = cv2.bilateralFilter(blur, 2, 20, 20)
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)       
        absX = cv2.convertScaleAbs(sobelx)
        absY = cv2.convertScaleAbs(sobely)
        dst = cv2.addWeighted( absX, 0.5, absY, 0.5,0)
        cv2.imshow('Result', dst)
        return dst        
        
    def 
        
    
    
    