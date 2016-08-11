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
from modelfitter import ModelFitter
#import Mouse_loc

mf1=ModelFitter()
mf1.add_graph()
mf1.image_fitting(loc,8,6,40)

# Automatic initialization
image = cv2.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/extra/20.tif')
t11 = Locator()
loc2=t11._obj_dec_(image,40,400,400)
clone = image.copy()
cv2.circle(clone, (loc2[0],loc2[1]),50,255,-1)
small = cv2.resize(clone, (0,0), fx=0.3, fy=0.3)
cv2.imshow("Window", small)
mf1.image_fitting(loc2,8,6,40)


