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
           # Function return an object Teeth t_0 ***
             t_mean = Teeth(0)
             t_mean.Teeth = np.zeros((3200,2))         
             for i in range(len(self.Patients)):
                    t_mean.Teeth = t_mean.Teeth + self.Patients[i].Teeth
             t_mean.Teeth  =t_mean.Teeth / len(Patients)
             return t_mean if len(Patients) != 0 else t_mean      
               
       def _procrustes_analysis(self, Patients):
           # Return a aligned model with all Teeth within this model converged.
             initial_mean_shape = self._get_mean_shape(self.Patients)
             self._weight_matrix(Patients)
             self.Patients[1:] = [t.align_to_shape(self.Patients[0], self.weight_matrix_) for t in self.Patients[1:]]
             _init_diff = initial_mean_shape.Teeth - self._get_mean_shape(self.Patients).Teeth
             # Inspired by https://github.com/CMThF/MIA_ActiveShapeModel/blob/master/generalProcrustes.m
             # For efficiency
             diff_token = np.sum(np.sum(abs(_init_diff)))
             ratio = 0
             while ratio < 0.98:
                 _mean_shape = self._get_mean_shape(self.Patients)
                 self._weight_matrix(self.Patients)
                 self.Patients[:] = [t.align_to_shape(_mean_shape, self.weight_matrix_) for t in self.Patients]
                 _mean_shape_after = self._get_mean_shape(self.Patients)
                 diff_token_after = np.sum(np.sum(abs(_mean_shape.Teeth - _mean_shape_after.Teeth)))
                 ratio = diff_token_after / diff_token
                 diff_token = diff_token_after
            
       def _PCA(self, Patients):
               if not Patients:
                    raise RuntimeError('There is no patients in this model!')
               #num_points = 3200  
               num_patients = len(self.Patients)   
               _all_teeths = np.zeros(shape=(num_patients,6400))
               for i,t in enumerate(self.Patients):
                   _all_teeths[i,:] = np.ravel(t.Teeth)
               # Use inalg.eig to do eigenvalue decomposition. 
               # Inspired from https://github.com/andrewrch/active_shape_models/blob/master/active_shape_models.py
               cov = np.cov(_all_teeths, rowvar=0)
               evals, evecs = np.linalg.eig(cov)
               evals = evals.real
               evecs = evecs.real
               ratio = np.divide(evals,sum(evals))
               _evals = evals[:len(ratio[np.cumsum(ratio)<0.99])]
               _evecs = evecs[:len(_evals)]
               return (_evals, _evecs,len(_evals)) 
 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
       def _weight_matrix(self,Patients):  
               # Inspired by https://github.com/andrewrch/active_shape_models
               self.Patients = Patients
               if not Patients:
                    return np.array([0])
               #num_points = 3200  
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
                
                   
                   
                                                                                                                                                                                 
               
            
           
       #def _weight_matrix(self,Patients):
           # Inspired by https://github.com/andrewrch/active_shape_models
          # self.Patients = Patients 
           #number of points on each unique shape
          # if not Patients:
                #return np.array([0])
           #num_points = 3200
           #weight_matrix = np.zeros(shape=(len(self.Patients),num_points,num_points))
           #for i in range(int(weight_matrix.shape[0])):
               #for k in range(num_points):
                   #for l in range(num_points):
                       #weight_matrix[i,k,l] = math.sqrt((self.Patients[i].Teeth[l,0] - self.Patients[i].Teeth[k,0])**2 + (self.Patients[i].Teeth[l,1] - self.Patients[i].Teeth[k,1])**2)
                                                         
           #w = np.zeros(num_points)
           #for k in range(num_points):
               #for l in range(num_points):
                  # w[k] += np.var(weight_matrix[:,k,l])
           #self.weight_matrix_ =  1/w
           