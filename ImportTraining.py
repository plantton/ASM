import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas


# Clear all variables
sys.modules[__name__].__dict__.clear()
# Close all opened windows
plt.close("all")

# Get the current workin directory
ASMdir = os.getcwd()

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
      tLdMat[ldlist.index(i),:] = np.loadtxt(i)   
      
# Transfer tLdMat into a dataframe, then add row names as index
tLdMat = pandas.DataFrame(tLdMat, index=ldlist)



