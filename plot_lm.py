# plot landmarks test

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import Akima1DInterpolator

# Currently only for interpolation of one teeth.
def interpolate_teeth(array, nInterp,verbose):
    temp = array[row_index,]
    temp_x = temp[0:-1:2]
    temp_y = temp[1:-1:2]
    temp_y = np.append(temp_y,temp[-1])
    plt.plot(temp_x,temp_y,'ro')
    plt.show()
    # Use itertion to interpolate points between landmarks
    # 40 is the number of landmarks for one teeth
    totalx = np.zeros(shape=(40*(nInterp-1)))
    totaly = np.zeros(shape=(40*(nInterp-1)))
    
    for i in range(len(temp_x)):
         # Interpolate $nInterp$ points between two closest landmarks
        temp_x_interp = np.linspace(temp_x[i-1], temp_x[i], num=nInterp)            
        if temp_x[i-1]-temp_x[i] < 0:
            bi = Akima1DInterpolator([temp_x[i-1], temp_x[i]], [temp_y[i-1], temp_y[i]])
            temp_y_interp = bi(temp_x_interp)
        if temp_x[i-1]-temp_x[i] == 0:
            temp_y_interp = np.linspace(temp_y[i-1], temp_y[i], num=nInterp)
        if temp_x[i-1]-temp_x[i] > 0:
            bi = Akima1DInterpolator([temp_x[i], temp_x[i-1]], [temp_y[i], temp_y[i-1]])
            temp_y_interp_reversed = bi(temp_x_interp[::-1])
            temp_y_interp = temp_y_interp_reversed[::-1]
                
        totalx[(i*(nInterp-1)):((i+1)*(nInterp-1))] = temp_x_interp[0:-1]
        totaly[(i*(nInterp-1)):((i+1)*(nInterp-1))] = temp_y_interp[0:-1]
        
    Vertices = np.zeros((1200,2))
    Vertices[:,0] = totalx
    Vertices[:,1] = totaly