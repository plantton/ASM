import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas
import math

def Aligndata(Vertices):
    # Centralisation
    offsetv = -np.mean(Vertices, axis=0)
    Vertices[:,0] = Vertices[:,0] + offsetv[0];
    Vertices[:,1] = Vertices[:,1] + offsetv[1];
    
    # Remove all rotations
    rot = math.atan2(Vertices[:,1],Vertices[:,0]);
    
    