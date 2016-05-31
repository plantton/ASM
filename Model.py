import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas
from Teeth import Teeth
import math

# Active Shape Model
# Inspired by https://github.com/andrewrch/active_shape_models
class Model: 
       
       ASMdir = 'C:/Users/tangc/Documents/ComVi'
       work_path = ASMdir + '/ASM'
       
       def __init__(self,Patients = []):
          self.Patients = Patients
          self.weight_matrix_ = self._weight_matrix(Patients)
       
       def _get_patients(self,i):
            if i in range(1,15,1):
              for i in range(1,i+1,1):
                token = Teeth(i)
                token.create_teeth()
                self.Patients.append(token)
            else:
                raise RuntimeError('Patient number not in our set!')
                           
       def _get_mean_shape(self, Patients):
             _mean_shape =  np.zeros(shape=(3200,2))            
             for i in range(len(self.Patients)):
                    _mean_shape = _mean_shape + Patients[i].Teeth
             return _mean_shape / len(Patients) if len(Patients) != 0 else _mean_shape      
               
       def _procrustes_analysis(self, Patients):
             initial_shape = self.Patients[0].Teeth
             initial_mean_shape = self._get_mean_shape(self.Patients)
             for i,t in enumerate(self.Patients[1:]):
                 token = t.align_to_shape(initial_shape, self.weight_matrix_)
                 self.Patients[i] = token
             _init_diff = initial_mean_shape - self._get_mean_shape(self.Patients)
             
                
                                                                                                
                                                                                                                                               
       def _weight_matrix_rev(self,Patients):  
               # Inspired by https://github.com/andrewrch/active_shape_models
               self.Patients = Patients
               if not Patients:
                    return np.array([0])
               num_points = 3200  
               num_patients = len(self.Patients)   
               _all_teeths = np.zeros(shape=(3200,2*num_patients))
               for i,t in enumerate(self.Patients):
                   _all_teeths[:,i*2:2*i+2] = t.Teeth
               _tile_all_teeth = np.array([_all_teeths,]*3200)
               _dist_sqrt_pts = (_tile_all_teeth - np.transpose(_tile_all_teeth, (1, 0, 2)))**2
               _dist_sqrt_pts = np.sqrt(_dist_sqrt_pts[:, :,0::2] + _dist_sqrt_pts[:, :,1::2])
               _var_mat = np.var(_dist_sqrt_pts,axis=2)
               w = np.sum(_var_mat,axis=1)
               self.weight_matrix_ = 1/w
                
                   
                   
                                                                                                                                                                                 
               
            
           
       def _weight_matrix(self,Patients):
           # Inspired by https://github.com/andrewrch/active_shape_models
           self.Patients = Patients 
           #number of points on each unique shape
           if not Patients:
                return np.array([0])
           num_points = 3200
           weight_matrix = np.zeros(shape=(len(self.Patients),num_points,num_points))
           for i in range(int(weight_matrix.shape[0])):
               for k in range(num_points):
                   for l in range(num_points):
                       weight_matrix[i,k,l] = math.sqrt((self.Patients[i].Teeth[l,0] - self.Patients[i].Teeth[k,0])**2 + (self.Patients[i].Teeth[l,1] - self.Patients[i].Teeth[k,1])**2)
                                                         
           w = np.zeros(num_points)
           for k in range(num_points):
               for l in range(num_points):
                   w[k] += np.var(weight_matrix[:,k,l])
           self.weight_matrix_ =  1/w
           