import pandas
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

# A class represents 8 teeth for a certain patient.

class Teeth:
      ASMdir = 'C:/Users/tangc/Documents/ComVi'
      lddir = ASMdir+'/_Data/Landmarks/original/'
      
      def __init__(self, i, Teeth = np.zeros(shape=(3200,2))):
             self.name = 'Patient: '+str(i)
             self.Teeth = Teeth
             
      def _name_(self,i):
            if i in range(1,16,1):
                self.name = 'Patient: '+str(i)
            else:
                raise RuntimeError('Patient number not in our set!')

      # range of i is between 1 to 14.
      def create_teeth(self):
          i = int(self.name.split(':')[1])
          os.chdir(self.lddir)
          ld = 40;
          tLdMat = np.zeros(shape=(8,ld*2))
          ldlist = os.listdir(os.getcwd())
          idx = []
          for j, str_j in enumerate(ldlist):
                 if str_j.endswith(".txt") and int(str_j[9:-4].split('-')[0]) == i :
                    tLdMat[int(str_j[9:-4].split('-')[1])-1,:] = np.loadtxt(str_j)
                    idx.append(str_j)   
          tLdMat = pandas.DataFrame(tLdMat, index=idx)
          Teeth = np.zeros(shape=(3200,2))
          # Now interpolate the teeth and combine the eight teeth into one structure.          
          for l in range(tLdMat.shape[0]):
               tV = self.interpolate_teeth(tLdMat,idx,l, 11,False)
               Teeth[l*400:(l+1)*400,:] = tV
          self.Teeth = Teeth
      

      
      def interpolate_teeth(self,dataframe,labellist,j, nInterp,verbose):
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
           Vertices = np.zeros((400,2))
           Vertices[:,0] = totalx
           Vertices[:,1] = totaly
           return Vertices
       
      #  Plot the interpolated images on the original radiograph
      def show_graph(self):
           graph_dir = self.ASMdir+'/_Data/Radiographs'
           os.chdir(graph_dir)
           if len(str(int(self.name.split(':')[1]))) == 2:
               img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/'+str(int(self.name.split(':')[1]))+'.tif')
           else:
               img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/0'+str(int(self.name.split(':')[1]))+'.tif')
           fig = plt.figure()
           plt.imshow(img)
           plt.title('Patient ' + str(int(self.name.split(':')[1])))
           plt.plot(self.Teeth[:,0],self.Teeth[:,1],'g.',markersize=1.5)
           
      # Vectorize the following methods     
      def _num_points(self):
          return int(self.Teeth.shape[0])
          
      def _get_X(self, weight_matrix_):
            return sum(weight_matrix_ * self.Teeth[:,0])
            
      def _get_Y(self, weight_matrix_):
            return sum(weight_matrix_ * self.Teeth[:,1])
            
      def _get_Z(self, weight_matrix_):
            return sum(weight_matrix_ * (self.Teeth[:,0]**2+self.Teeth[:,1]**2))
            
      def _get_C1(self, weight_matrix_, T):
             return sum(weight_matrix_ * (self.Teeth[:,0]*T.Teeth[:,0] + self.Teeth[:,1]*T.Teeth[:,1]))
             
      def _get_C2(self, weight_matrix_, T):
             return sum(weight_matrix_ * (T.Teeth[:,1]*self.Teeth[:,0] - T.Teeth[:,0]*self.Teeth[:,1]))
             
      
                       
      def _alignment_parameters(self, T,weight_matrix_):
          # Inspired by https://github.com/andrewrch/active_shape_models
          # Based on the original functions on paper.
              X1 = T._get_X(weight_matrix_)
              X2 = self._get_X(weight_matrix_)
              Y1 = T._get_Y(weight_matrix_)
              Y2 = self._get_Y(weight_matrix_)
              Z = self._get_Z(weight_matrix_)
              W = sum(weight_matrix_)
              C1 = self._get_C1(weight_matrix_, T)
              C2 = self._get_C2(weight_matrix_, T)
              
              # Matrix in the original paper
              a = np.array([[ X2, -Y2,   W,  0],
                            [ Y2,  X2,   0,  W],
                            [  Z,   0,  X2, Y2],
                            [  0,   Z, -Y2, X2]])
                            
              b = np.array([X1, Y1, C1, C2])
              return np.linalg.solve(a, b)
              
      def _apply_new_model(self, para):
              token = np.zeros(shape=(3200,2))
              token[:,0] = (para[0]*self.Teeth[:,0] - para[1]*self.Teeth[:,1]) + para[2]
              token[:,1] = (para[1]*self.Teeth[:,0] + para[0]*self.Teeth[:,1]) + para[3]
              self.Teeth = token
              return self
              
      def align_to_shape(self, T, weight_matrix_):
          # Inspired by https://github.com/andrewrch/active_shape_models
              para = self._alignment_parameters(T,weight_matrix_)
              return self._apply_new_model(para)
              
      def get_normal_to_point(self, p_num):
              x = 0; y = 0; mag = 0
              if p_num <0 and p_num > self.Teeth.shape[0]:
                        raise RuntimeError('Point is out of range!')
              if p_num == 0:
                   x = self.Teeth[1,0] - self.Teeth[0,0]
                   y = self.Teeth[1,1] - self.Teeth[0,1] 
              elif p_num == self.Teeth.shape[0] - 1:
                   x = self.Teeth[-1,0] - self.Teeth[-2,0]
                   y = self.Teeth[-1,1] - self.Teeth[-2,1]
              else:
                   x = self.Teeth[p_num+1,0] - self.Teeth[p_num-1,0]
                   y = self.Teeth[p_num+1,1] - self.Teeth[p_num-1,1]
              mag = math.sqrt(x**2 + y**2)
              return (-y/mag, x/mag)
            
              
              
              
           


      
       

        
