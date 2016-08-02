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

class ModelFitter:
    file_in = 'C:/Userstangc/Documents/ComVi/_Data/Radiographs/extra/15.tif'
    
    def __init__(self, loc):
        # Loc is the initial location of Active Shape Model
        # Loc can be given by class Locator (automatic initialisation)
        # Loc can also be given by manual initialization (mouse input)
        self.loc = loc
        #file_in = image
        self.img = []
        self.model = Model()
        self.model._get_patients(14,8)
    
    def add_graph(self):
        #
        img = cv2.imread(file_in)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Clahe parameters were adjusted manually.
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(25,25))
        self.img = clahe.apply(gray)
        
    def init_shape(self):
        init_shape = self.model._get_mean_shape(self.model.Patients)
        return init_shape
         
         
        
        
   
               
    #def __produce_gradient_image(self):
    #    gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
    #    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(25,25))
    #    cl1=clahe.apply(gray)
    #    blur = cv2.GaussianBlur(cl1,(3,3),0)
    #    blur = cv2.bilateralFilter(blur, 2, 20, 20)
    #    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    #    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)       
    #    absX = cv2.convertScaleAbs(sobelx)
    #    absY = cv2.convertScaleAbs(sobely)
    #    dst = cv2.addWeighted( absX, 0.5, absY, 0.5,0)
    #    cv2.imshow('Result', dst)
    #    return dst        
        
    def image_fitting(self,k,ns,loop_num):
        # loop_num is the loop number for fitting process
        _evals_shape, _evecs_shape,_num_shape = self.model._PCA(self.Patients)
        _grey_mean,_grey_evals,_grey_evecs = self.model.greyscale_PCA(k)
        _init_shape = self.init_shape()
        init_mean_Teeth = _init_shape.Teeth
        for i in range(loop_num):
            _init_shape.get_normal_to_teeth()
            _Normals = _init_shape.Normals
            l = k + ns
            _init_shape.get_profile_and_Derivatives(k)
            f=np.zeros(ns*2+1,3200)
            for j in range(3200):
                for n in range(ns*2+1):
                   gi=np.zeros(2*k+1,1)
                   gi = _init_shape.profiles[n:2*k+n,j]
                   bc = _grey_evecs[j]*(gi-_grey_mean)
                   bc = bc/np.sqrt(_grey_evals[j])
                   f[n,j]=np.sum(bc**2)
            temp = np.amin(f, axis=0)
            _idx = np.argmin(f, axis=0)
            movement = (_idx-1)-ns
            _init_shape.Teeth[:,0] = _init_shape.Teeth[:,0] + movement*_Normals[:,0]
            _init_shape.Teeth[:,1] = _init_shape.Teeth[:,1] + movement*_Normals[:10] 
            x_search = np.ravel(_init_shape.Teeth)
            b = _evecs_shape*(x_search - np.ravel(init_mean_Teeth))
            # Suitable limits for the shape parameters; see paper.
            maxb = 3*np.sqrt(_evals_shape)
            b=np.max(np.array(np.amin(np.array(b,maxb),axis=0),-maxb),axis=0)
            #
            x_search =  np.ravel(init_mean_Teeth) + _evecs_shape.T*b
            _init_shape.Teeth = np.reshape(x_search,(_init_shape.Teeth.shape))
            
            
            
            #np.amin(a, axis=0)
                
                
                
                
                
                
                
                
                
                
                
        
    
    
    