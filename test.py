# Class Align Teeth image
import cv2
import cv2.cv as cv
import os
import numpy as np
import time
import fnmatch
import sys
# Package imutils installed via extra command
import imutils
from PIL import Image       



        def pyramid(image, scale=1.5, minSize=(30, 30)):
                # http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
   	        # yield the original image
           	yield image 
	        # keep looping over the pyramid
           	while True:
          		# compute the new dimensions of the image and resize it
          		w = int(image.shape[1] / scale)
          		image = imutils.resize(image, width=w)
            
          		# if the resized image does not meet the supplied minimum
          		# size, then stop constructing the pyramid
          		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
         			break            
          		# yield the next image in the pyramid
          		yield image
          		
        def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	      for y in xrange(image.shape[0]/3, image.shape[0], stepSize):
		for x in xrange(image.shape[1]/3, image.shape[1]*2/3, stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]]) 
		
        image = cv2.imread('C:/Users/tangc/Documents/ComVi/ASM/test.jpg')
        #C:\Users\tangc\Documents\ComVi\_Data\Radiographs
        image = cv2.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/03.tif')
        directory = 'C:/Users/tangc/Documents/ComVi/ASM/eigenteeth2'
        (winW, winH) = (100, 100)
        
        t11._obj_dec_(image,40,400,400)
        
        t11 = Locator()
        
        ori_pro = np.zeros((eigenvectors.shape[1], X.shape[0]))
        for i in range(X.shape[0]):
            ori_pro[:,i] = t11.project(eigenvectors,X[i,:],mu)

        X = t11.createX(directory)
        [eigenvalues, eigenvectors, mu] = t11.pca(X,10)
        W_p = t11.project(eigenvectors,X[3,:],mu)
        W_rep = np.reshape(np.tile(W_p, (1,14)),(14,10)).T
        W_rep.shape
        
        W=np.zeros(shape=[1,winW*winH])
        #img = cv2.imread(window)
        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        W[0,:] = gray.flatten()
        W_p = self.project(eigenvectors,W,mu)
        W_rep = np.matlib.repmat(W_p,1,14)
        
        image = cv2.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/08.tif')
        loc = t11._obj_dec_(image,40,400,400)
        clone = image.copy()
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.circle(clone, (loc[0],loc[1]),50,255,-1)
        small = cv2.resize(clone, (0,0), fx=0.3, fy=0.3)
        #plt.imshow(clone)
        #plt.Circle((1407, 1172),500,color='r')
        #plt.show()
        
        cv2.imshow("Window", small)