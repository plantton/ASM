# plot landmarks test

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from PIL import Image

# Currently only for interpolation of one teeth.
def interpolate_teeth(dataframe,labellist,j, nInterp,verbose):
    temp = np.asarray(dataframe.loc[[labellist[j]]])
    temp = np.ravel(temp)
    temp_x = temp[0:-1:2]
    temp_y = temp[1:-1:2]
    temp_y = np.append(temp_y,temp[-1])
    if bool(verbose):
        if len(labellist[j][9:-4].split('-')[0]) == 2:
            img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/'+labellist[j][9:-4].split('-')[0]+'.tif')
        else:
            img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/0'+labellist[j][9:-4].split('-')[0]+'.tif')
        plt.imshow(img)
        plt.title('Patient ' + labellist[j][9:-4])
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
    
    if bool(verbose):
        plt.plot(totalx,totaly,'g-')
    # Vertices    
    Vertices = np.zeros((1200,2))
    Vertices[:,0] = totalx
    Vertices[:,1] = totaly
    
    # Lines
    Lines = np.zeros(Vertices.shape)
    Lines[:,0] = np.arange(0,Vertices.shape[0],1)
    Lines[0:Lines.shape[0]-1,1] = np.arange(1,Vertices.shape[0],1)
    Lines[-1,1] = 0
    
    return Vertices,Lines
    