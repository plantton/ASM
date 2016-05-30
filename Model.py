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
                           
       def _weight_matrix(self,Patients):
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
           