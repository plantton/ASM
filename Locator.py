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
import numpy.matlib
from PIL import Image
import matplotlib.pyplot as plt

class Locator:

        def create_database(directory, show = True):
            # Resize all training images into the same size 
            for filename in fnmatch.filter(os.listdir(directory),'*.jpg'):
                    file_in = directory+"/"+filename
                    file_out= directory+"2/"+filename
                    img = cv2.imread(file_in)
                    if show:
                        vis = img.copy()
                        cv2.imshow('img', vis)
                        cv2.waitKey(0)
                    result = cv2.resize(vis, (400,400))
                    if show:
                        cv2.imshow('img', result)
                        cv2.waitKey(0)            
                    cv2.imwrite(file_out, result)
            cv2.destroyAllWindows()



        def createX(self,directory,nbDim=160000):
            nbImages = len(os.listdir(directory))
            X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
            
            for i,filename in enumerate( fnmatch.filter(os.listdir(directory),'*.jpg') ):
                file_in = directory+"/"+filename
                img = cv2.imread(file_in)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                X[i,:] = gray.flatten()
            return X

        def project(self,W, X, mu):
            if mu is None:
                return np.dot(X,W)
            return np.dot(X - mu, W)

        def reconstruct(self,W, Y, mu):
            if mu is None:
                return np.dot(Y,W.T)
            return np.dot(Y, W.T) + mu

        def pca(self,X, num_components):
            [n,d] = X.shape
            if (num_components <= 0) or (num_components>n):
                num_components = n
            mu = X.mean(axis=0)
            X = X - mu
            if n>d:
                C = np.dot(X.T,X)
                [eigenvalues,eigenvectors] = np.linalg.eigh(C)
            else:
                C = np.dot(X,X.T)
                [eigenvalues,eigenvectors] = np.linalg.eigh(C)
                eigenvectors = np.dot(X.T,eigenvectors)
                for i in xrange(n):
                    eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
            # or simply perform an economy size decomposition
            # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
            # sort eigenvectors descending by their eigenvalue
            idx = np.argsort(-eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:,idx]
            # select only num_components
            eigenvalues = eigenvalues[0:num_components].copy()
            eigenvectors = eigenvectors[:,0:num_components].copy()
            return [eigenvalues, eigenvectors, mu]
        
            
        #def pyramid(image, scale=1.5, minSize=(30, 30)):
        #        # http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
   	    #    # yield the original image
        #   	yield image 
	       # # keep looping over the pyramid
        #   	while True:
        #  		# compute the new dimensions of the image and resize it
        #  		w = int(image.shape[1] / scale)
        #  		image = imutils.resize(image, width=w)
        #    
        #  		# if the resized image does not meet the supplied minimum
        #  		# size, then stop constructing the pyramid
        #  		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        # 			break            
        #  		# yield the next image in the pyramid
        #  		yield image
		            
 #       def sliding_window(self,image, stepSize, windowSize):
	## slide a window across the image
	#      for y in xrange(0, image.shape[0], stepSize):
	#	for x in xrange(0, image.shape[1], stepSize):
	#		# yield the current window
	#		yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])    
        
        def _sliding_window(self,image, stepSize, windowSize):
	# slide a window across part of the image
	      for y in xrange(image.shape[0]/3, (image.shape[0]-400), stepSize):
		for x in xrange(image.shape[1]/3, (image.shape[1]*2/3)-400, stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]]) 
		       
        
        def _obj_dec_(self,image,stepSize,winW,winH):
            # loop over the image pyramid
            X = self.createX("C:/Users/tangc/Documents/ComVi/ASM/eigenteeth2")
            [eigenvalues, eigenvectors, mu] = self.pca(X,num_components=10)
            ori_pro = np.zeros((eigenvectors.shape[1], X.shape[0]))
            for i in range(X.shape[0]):
                ori_pro[:,i] = self.project(eigenvectors,X[i,:],mu)
            #i = 0
            token = sys.float_info.max
            ls = []
            ls.append([0,0])
            #for resized in self.pyramid(image):
                  # Record the number of this image in the pyramid
                #i += 1
                  # loop over the sliding window for each layer of the pyramid
                  # token = sys.float_info.max
            for (x, y, window) in self._sliding_window(image, stepSize, windowSize=(winW, winH)):
      		# if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
     			continue         		
                W=np.zeros(shape=[1,winW*winH])
                #img = cv2.imread(window)
                gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                W[0,:] = gray.flatten()
                W_p = self.project(eigenvectors,W,mu)
                W_rep = np.reshape(np.tile(W_p, (1,14)),(14,10)).T
                # Projection page 7 on the paper
                #fi_f = np.dot(eigenvectors,eigenvalues)
                epsilon_ = np.sum(np.linalg.norm(W_rep - ori_pro, axis=0))
                #epsilon_ = np.linalg.norm(fi_ - fi_f)
                if epsilon_ < token:
                    ls[-1] = [x,y]
                token = epsilon_
  		# since we do not have a classifier, we'll just draw the window
      		clone = image.copy()
      		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
      		#cv2.circle(clone, (1407, 1172),50,255,-1)
      		small = cv2.resize(clone, (0,0), fx=0.3, fy=0.3)
      	        #plt.imshow(clone)
                #plt.Circle((1407, 1172),500,color='r')
                #plt.show()
      		 
      		cv2.imshow("Window", small)
      		cv2.waitKey(1)
      		time.sleep(0.025)
            #ls.append([0,0,0])
            loc = np.array([ls[0][0],ls[0][1]-230])
            #scale=1.5
            #_w = int(image.shape[1] / (scale**loc[0]))
            #x_o = image.shape[1] * (loc[1]/_w)
            #y_o = image.shape[0] * (loc[1]/_w)
            # Round ???
            return loc
            
    
