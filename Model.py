import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas
import Teeth
import math

# Active Shape Model
# Inspired by https://github.com/andrewrch/active_shape_models
class Model: 
       
       ASMdir = 'C:/Users/tangc/Documents/ComVi'
       work_path = ASMdir + '/ASM'
       
       def __init__(self,Teeth):
          self.Teeth = Teeth
       
       # Create an array containing a specified size of training data.
       def create_training(i):
           training_array = np.zeros(shape=(3200,i*2))
           for j in range(i):
               training_array[:,[2*j,2*j+1]] = create_teeth(j+1)
           return training_array
           
       def weight_matrix(training_array):
           #number of points on each unique shape
           num_points = int(training_array.shape[0])
           weight_matrix = np.zeros(shape=(int(training_array.shape[1]/2),num_points,num_points))
           for i in range(int(weight_matrix.shape[0])):
               for k in range(num_points):
                   for l in range(num_points):
                       weight_matrix[i,k,l] = math.sqrt((training_array[l,i*2] - training_array[k,i*2])**2 + (training_array[l,i*2+1] - training_array[k,i*2+1])**2)
                       
           w = np.zeros(num_points)
           for k in range(num_points):
               for l in range(num_points):
                   w[k] += np.var(weight_matrix[:,k,l])
           return 1/w
               
            
                       
           
           