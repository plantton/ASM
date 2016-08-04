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
import pickle


# A class to fit ASM to a given image
# Greyscale profile along normal direction of landmarks

class ModelFitter:
    file_in = 'C:/Users/tangc/Documents/ComVi/_Data/Radiographs/extra/16.tif'
    
    def __init__(self):
        # Loc is the initial location of Active Shape Model
        # Loc can be given by class Locator (automatic initialisation)
        # Loc can also be given by manual initialization (mouse input)
        self.loc = []
        #file_in = image
        self.img = []
        self.model = Model()
        self.model._get_patients(14,8)
    
    def add_graph(self):
        #
        img = cv2.imread(self.file_in)
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
        img = cv2.imread(self.file_in)
        clone = img.copy()
        cv2.namedWindow("Fitting Window", cv2.WINDOW_AUTOSIZE)
        small = cv2.resize(clone, (0,0), fx=0.3, fy=0.3)
        cv2.imshow( "Fitting Window", small )
        cv2.waitKey(1)
        os.chdir("C:/Users/tangc/Documents/ComVi/ASM")
        shape_datas = pickle.load( open( "save.p", "rb" ) )
        _evals_shape = shape_datas[0]
        _evecs_shape = shape_datas[1]
        #_num_shape = shape_datas[0]
        _grey_mean,_grey_evals,_grey_evecs = self.model.greyscale_PCA(k)
        _init_shape = self.init_shape()
        init_mean_Teeth = _init_shape.Teeth
        for i in range(loop_num):
            _init_shape.get_normal_to_teeth()
            _Normals = _init_shape.Normals
            l = k + ns
            _init_shape.alter_get_profile_and_Derivatives(l,self.file_in)
            f=np.zeros(shape=(ns*2+1,3200))
            for j in range(3200):
                for n in range(ns*2+1):
                   gi=np.zeros(shape=(2*k+1,1))
                   gi = _init_shape.profiles[n:17+n,j]
                   bc = np.dot(_grey_evecs[j],(gi-_grey_mean[j]))
                   bc = bc/np.sqrt(_grey_evals[j])
                   f[n,j]=np.sum(bc**2)
            temp = np.amin(f, axis=0)
            _idx = np.argmin(f, axis=0)
            movement = (_idx-1)-ns
            _init_shape.Teeth[:,0] = _init_shape.Teeth[:,0] + movement*_Normals[:,0]
            _init_shape.Teeth[:,1] = _init_shape.Teeth[:,1] + movement*_Normals[:,1]
            clone = img.copy()
            for s in range(_init_shape.Teeth.shape[0]):
                _pointx = _init_shape.Teeth[s,0].astype('int')
                _pointy = _init_shape.Teeth[s,1].astype('int')
                cv2.circle(clone,(_pointx,_pointy),3,(0,255,0),1,8,3)
            small = cv2.resize(clone, (0,0), fx=0.3, fy=0.3)
            cv2.imshow( "Fitting Window", small ) 
            x_search = np.ravel(_init_shape.Teeth)
            b = np.dot(_evecs_shape,(x_search - np.ravel(init_mean_Teeth)))
            # Suitable limits for the shape parameters; see paper.
            maxb = 3*np.sqrt(_evals_shape)
            b=np.max(np.array([np.amin(np.array([b,maxb]),axis=0),-maxb]),axis=0)
            #
            x_search =  np.ravel(init_mean_Teeth) + np.dot(_evecs_shape.T,b)
            _init_shape.Teeth = np.reshape(x_search,(_init_shape.Teeth.shape))
      	    cv2.waitKey(10)
      	    #time.sleep(2.0)
      	
      	
      	 
            
            #np.amin(a, axis=0)
                
                
                
                
                
                
                
                
                
                
                
        
    
    
    