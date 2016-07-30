# -*- coding: utf-8 -*-
import pandas
import math
import os
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
import cv2
from scipy import interpolate

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
           
      def __self_image(self):
            graph_dir = self.ASMdir+'/_Data/Radiographs'
            os.chdir(graph_dir)
            if len(str(int(self.name.split(':')[1]))) == 2:
                img = cv2.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/'+str(int(self.name.split(':')[1]))+'.tif')
            else:
                img = cv2.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/0'+str(int(self.name.split(':')[1]))+'.tif')
            return img
           
           
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
              
      def __get_normal_to_point(teeth_array, p_num):
              x = 0; y = 0; mag = 0
              if p_num <0 and p_num >teeth_array.shape[0]-1:
                        raise RuntimeError('Point is out of range!')
              if p_num == 0:
                   x = teeth_array[1,0] - teeth_array[-1,0]
                   y = teeth_array[1,1] - teeth_array[-1,1] 
              elif p_num == teeth_array.shape[0] - 1:
                   x = teeth_array[0,0] - teeth_array[-2,0]
                   y = teeth_array[0,1] - teeth_array[-2,1]
              else:
                   x = teeth_array[p_num+1,0] - teeth_array[p_num-1,0]
                   y = teeth_array[p_num+1,1] - teeth_array[p_num-1,1]
              mag = math.sqrt(x**2 + y**2)
              # Return sin(α) and cos(α) in sequence
              # 'α' is the angle 
              return (abs(y)/mag, abs(x)/mag, x*y)
              
      # Vectorization of _get_normal method        
      def __get_normal_to_tooth(teeth_array):
              # teeth_array.shape = (400,2)
              T1=np.roll(teeth_array, -1, axis=0)
              T2=np.roll(teeth_array, 1, axis=0)
              TN = T1 - T2
              token = TN**2
              mag = token[:,0]+token[:,1]
              mag = sqrt(mag)
              token_N=np.divide(TN.T,mag).T
              token_N=np.roll(token_N, 1, axis=1)
              token_N[:,1]=np.negative(token_N[:,1])
              return token_N
              
      def __get_normal_to_teeth(self):
              teeth_normals=np.zeros(shape=self.Teeth)
              for l in range(8):
                    tV = self.Teeth[l*400:(l+1)*400,:]
                    token_N = self.__get_normal_to_tooth(tV)
                    teeth_normals[l*400:(l+1)*400,:] = token_N
              return teeth_normals
      
      # Inspired from MATLAB AAM codes, the most efficient vectorization method
      def __linspace_multi(array_1,array_2,num_profile):
               mat1=array([array_1,]*(num_profile-1)).transpose() 
               mat2=array([range(num_profile-1),]*array_1.shape[0]) 
               mat3=array([(array_2-array_1),]*(num_profile-1)).transpose()
               mat3=mat3/(num_profile-1)
               # Mat
               token_mat=mat1+mat2*mat3
               lin_mat = np.zeros(shape=(token_mat.shape[0],token_mat.shape[1]+1))
               lin_mat[:,0:-1]=token_mat
               lin_mat[:,-1]=array_2
               return lin_mat
               
       def
                        
              
              
              
      #def __get_profilepoints(self,teeth_array,p_num,k):
      #          # p_num = [0,399]
      #          # 'k' is the number of points on each sides of the landmark
      #          # Return an array contains the coordinates of the points on the profile
      #          _sinA, _cosA, _xy = self.__get_normal_to_point(teeth_array,p_num)
      #          # Both _x and _y are positive values
      #          #_x = _sinA*k
      #          #_y = _cosA*k
      #          normal_profile = np.zeros(shape=(2*k+1,2))
      #          normal_profile[k,:] = teeth_array[p_num,:]
      #          # A littile bit proof of geometry on the paper :)
      #          if _xy >= 0:
      #              for i in range(k):
      #                  normal_profile[i,0] = teeth_array[p_num,0] - (k-i)*_sinA
      #                  normal_profile[i,1] = teeth_array[p_num,1] + (k-i)*_cosA
      #                  normal_profile[k+i+1,0] = teeth_array[p_num,0] + (i+1)*_sinA
      #                  normal_profile[k+i+1,1] = teeth_array[p_num,1] - (i+1)*_cosA
      #          else:
      #              for i in range(k):
      #                  normal_profile[i,0] = teeth_array[p_num,0] - (k-i)*_sinA
      #                  normal_profile[i,1] = teeth_array[p_num,1] - (k-i)*_cosA
      #                  normal_profile[k+i+1,0] = teeth_array[p_num,0] + (i+1)*_sinA
      #                  normal_profile[k+i+1,1] = teeth_array[p_num,1] + (i+1)*_cosA
      #          normal_profile = np.around(normal_profile)
      #          normal_profile = normal_profile.astype(int)
      #          return  normal_profile # The coordiantes shall be integers
                
    def __get_profilepoints(self,):
                
      #def __get_Normals(self):
      #          Lines = np.zeros(self.Teeth.shape)    
      #          Lines[:,0] = np.array(range(len(self.Teeth)))  
      #          Lines[:-1,1] = np.array(range(1,len(self.Teeth)))
      #          Lines = Lines.astype(int)
      #          DT = self.Teeth[Lines[:,0],:] - self.Teeth[Lines[:,1],:] 
      #          D1 = np.zeros(self.Teeth.shape)
      #          D2 = np.zeros(self.Teeth.shape)
      #          D1[Lines[:,0],:] = DT
      #          D2[Lines[:,1],:] = DT
      #          D=D1+D2
      #          L = np.sqrt(D[:,0]**2+D[:,1]**2)
      #          Normals = np.zeros(self.Teeth.shape)
      #          Normals[:,0] = np.divide(D[:,1], L)
      #          Normals[:,1] = np.divide(D[:,0], L)
      #          return Normals
                
      #def __linspace_multi(d1,d2,i):
      #          token = np.array([d1,]*(i-1)).transpose() + np.multiply(numpy.matlib.repmat(np.arange(i-1),len(d1),1),np.array([(d2-d1),]*(i-1)).transpose())/(math.floor(i)-1)
      #          result = np.zeros((token.shape[0],token.shape[1]+1))
      #          result[:,:-1] = token
      #          result[:,-1] = d2           
      #          return result
      #      
      #def __getProfileAndDerivatives2D(self,k):
      #          image = self.__self_image()
      #          gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      #          clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(25,25))
      #          image=clahe.apply(gray)
      #          k = 8
      #          gtc = np.zeros(shape=((k*2+1),len(self.Teeth)))
      #          dgtc = np.zeros(shape=((k*2+1),len(self.Teeth)))
      #          Normals = self.__get_Normals()
      #          #
      #          xi=__linspace_multi(m1.Patients[0].Teeth[:,0]-Normals[:,0]*k, m1.Patients[0].Teeth[:,0]+Normals[:,0]*k,k*2+1)
      #          yi=__linspace_multi(m1.Patients[0].Teeth[:,1]-Normals[:,1]*k, m1.Patients[0].Teeth[:,1]+Normals[:,1]*k,k*2+1)
      #          xi[xi < 1] = 1
      #          xi[xi > image.shape[0]] = image.shape[0]
      #          yi[yi < 1] = 1
      #          yi[yi > image.shape[1]] = image.shape[1]
      #          #
      #          y = np.arange(0,image.shape[0])
      #          x = np.arange(0,image.shape[1])
      #          f = interpolate.RectBivariateSpline(x,y,image.T)
      #          gt = (f.ev(xi,yi)).T
      #          gt[np.isnan(gt)] = 0
      #          dgt = np.zeros((17,3200))
      #          dgt[0,:] = gt[1,:]-gt[0,:]
      #          dgt[1:-1,:] = (gt[2:,:]-gt[:-2,:])/2
      #          dgt[-1,:] = gt[-1,:] - gt[-2,:]

