import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas

def Procruste_Weight(Vertices_1, Vertices_2):
    # Produce the weight matrix used in the training set alignment
    
    # 'w' is the weight matrix we are looking for.
    w = np.zeros(Vertices_1.shape[0])
    