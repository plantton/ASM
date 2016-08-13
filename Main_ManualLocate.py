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
import plotly.plotly as py
#import Mouse_loc

mf1=ModelFitter()
mf1.add_graph()
_init_shape = mf1.init_shape()
init_mean_Teeth = np.copy(_init_shape.Teeth)
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


# Show shape eigenvector variations
shape_datas = pickle.load( open( "save.p", "rb" ) )
plt.close('all')
#fig, ax = plt.subplots(2, 5)
for i in range(shape_datas[2]):
    token = np.ravel(init_mean_Teeth) + shape_datas[1][i,:]*np.sqrt(shape_datas[0][i])*3
    token = np.reshape(token,(3200,2))
    ax = plt.subplot(2,5,i+1)
    ax.plot(token[:,0],token[:,1],'g.',markersize=5)
    ax.plot(init_mean_Teeth[:,0],init_mean_Teeth[:,1],'r.',markersize=1.5)
    











