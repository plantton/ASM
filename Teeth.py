import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

# A class represents 8 teeth for a certain patient.

class Teeth:
      ASMdir = 'C:/Users/tangc/Documents/ComVi'
      lddir = ASMdir+'/_Data/Landmarks/original/'

      # range of i is between 1 to 14.
      def create_teeth(i):
          os.chdir(lddir)
          ld = 40;
          tLdMat = np.zeros(shape=(8,ld*2))
          ldlist = os.listdir(os.getcwd())
          idx = []
          for j, str_j in enumerate(ldlist):
                 if str_j.endswith(".txt") and int(str_j[9:-4].split('-')[0]) == i :
                    tLdMat[int(str_j[9:-4].split('-')[1])-1,:] = np.loadtxt(str_j)
                    idx.append(str_j)   
          tLdMat = pandas.DataFrame(tLdMat, index=idx)
          # Now interpolate the teeth and combine the eight teeth into one structure.
          Teeth = np.zeros(shape=(9600,2))
          for l in range(tLdMat.shape[0]):
               tV = interpolate_teeth(tLdMat,idx,l, 31,False)
               Teeth[l*1200:(l+1)*1200,:] = tV
          return Teeth
      
      
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
               plt.title('Patient ' + labellist[j][9:-4].split('-')[0])
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
           return Vertices
       
      #  Plot the interpolated images on the original radiograph
      def show_graph(i):
           graph_dir = ASMdir+'/_Data/Radiographs'
           Teeth_i = create_teeth(i)
           os.chdir(graph_dir)
           if len(str(i)) == 2:
               img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/'+str(i)+'.tif')
           else:
               img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/0'+str(i)+'.tif')
           fig = plt.figure()
           plt.imshow(img)
           plt.title('Patient ' + str(i))
           plt.plot(Teeth_i[:,0],Teeth_i[:,1],'g.',markersize=1.5)
           


      
       

        
