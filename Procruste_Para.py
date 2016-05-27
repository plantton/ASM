import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas

# Calculate the procruste parameters. Modified procruste method from literature - Cootes. ET AL.
def Procruste_Para(filename,nBetween,verbose): 
    
    
    # Interpolate to get more points
    
    
   