import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas
import Interpolate_teeth



# Clear all variables
#sys.modules[__name__].__dict__.clear()
# Close all opened windows
#plt.close("all")

# Get the current workin directory
ASMdir = 'C:/Users/tangc/Documents/ComVi'

# Change working dir to folder containing original landmarks
lddir = ASMdir+'/_Data/Landmarks/original/'
os.chdir(lddir)

# Create an empty array to store all landmarks from training dataset
# Landmark number ***
ld = 40;
tLdMat = np.zeros(shape=(len(os.listdir(os.getcwd())),ld*2))
# Iterate all landmark files, store all landmarks into a matrix
ldlist = os.listdir(os.getcwd())
for i in ldlist: 
     if i.endswith(".txt") :
      tLdMat[ldlist.index(i),:] = np.loadtxt(i)   
      
# Transfer tLdMat into a dataframe, then add row names as index
tLdMat = pandas.DataFrame(tLdMat, index=ldlist)

# Test for plot tooth
# Create a new window
for i in range(len(ldlist)):
  if i % 8 == 0:
    fig = plt.figure((i/8)+1)
    ax = fig.add_subplot(111, autoscale_on=True)
    if len(ldlist[i][9:-4].split('-')[0]) == 2:
        img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/'+ldlist[i][9:-4].split('-')[0]+'.tif')
    else:
        img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/0'+ldlist[i][9:-4].split('-')[0]+'.tif')
    plt.imshow(img)
    plt.title('Patient ' + ldlist[i][9:-4].split('-')[0])
    [Vi,Li] = interpolate_teeth(tLdMat,ldlist,i, 31,False)
    plt.plot(Vi[:,0],Vi[:,1],'g-')
    ax.annotate(ldlist[i][9:-4].split('-')[1], xy=(Vi[0,0], Vi[0,1]), xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
  else:
      if len(ldlist[i][9:-4].split('-')[0]) == 2:
        img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/'+ldlist[i][9:-4].split('-')[0]+'.tif')
      else:
        img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/0'+ldlist[i][9:-4].split('-')[0]+'.tif')
      plt.imshow(img)
      plt.title('Patient ' + ldlist[i][9:-4].split('-')[0])
      [Vi,Li] = interpolate_teeth(tLdMat,ldlist,i, 31,False)
      plt.plot(Vi[:,0],Vi[:,1],'g-')
      ax.annotate(ldlist[i][9:-4].split('-')[1], xy=(Vi[0,0], Vi[0,1]), xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
      



