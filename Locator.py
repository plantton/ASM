# Class Align Teeth image
import cv2
import cv2.cv as cv
import os
import numpy as np
import time
import fnmatch
# Package imutils installed via extra command
import imutils

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



        def createX(directory,nbDim=160000):
            nbImages = len(os.listdir(directory))
            X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
            
            for i,filename in enumerate( fnmatch.filter(os.listdir(directory),'*.jpg') ):
                file_in = directory+"/"+filename
                img = cv2.imread(file_in)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                X[i,:] = gray.flatten()
            return X

        def project(W, X, mu=None):
            if mu is None:
                return np.dot(X,W)
            return np.dot(X - mu, W)

        def reconstruct(W, Y, mu=None):
            if mu is None:
                return np.dot(Y,W.T)
            return np.dot(Y, W.T) + mu

        def pca(X, num_components=0):
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
	      for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])    
        
        
        def _obj_dec_(self,image,winW,winH):
            # loop over the image pyramid
            X = self.createX("C:/Users/tangc/Documents/ComVi/ASM/eigenteeth2")
            [eigenvalues, eigenvectors, mu] = self.pca(X,num_components=10)
            for resized in self.pyramid(image, scale=1.5):
       	    # loop over the sliding window for each layer of the pyramid
           	for (x, y, window) in self.sliding_window(resized, stepSize=4, windowSize=(winW, winH)):
          		# if the window does not meet our desired window size, ignore it
          		if window.shape[0] != winH or window.shape[1] != winW:
         			continue
            
          		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
          		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
          		# WINDOW
          		W=np.zeros(shape=[1,winW*winH])
          		img = cv2.imread(window)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        W[0,:] = gray.flatten()
                        fi = W - mu
                        # Projection page 7 on the paper
                        fi_f = self.project(eigenvectors, W, mu )
                        epsilon = np.linalg.norm(fi - fi_f)
          		# since we do not have a classifier, we'll just draw the window
          		clone = resized.copy()
          		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
          		cv2.imshow("Window", clone)
          		cv2.waitKey(1)
          		time.sleep(0.025)
  		
  		
        if __name__ == '__main__':
            #create database of normalized images
        #    for directory in ["data/arnold", "data/obama"]:
        #        create_database(directory, show = False)
            
            show = True
            
            #create big X arrays for arnold and obama
            X = createX("C:/Users/tangc/Documents/ComVi/ASM/eigenteeth2")
            #Call = ["arnold"]*Xa.shape[0] + ["obama"]*Xb.shape[0]
            
            nbImages = X.shape[0]
            #nbCorrect = 0
            for leave_out in range(nbImages):
                #remove leave_out from Xall
                select = np.ones(nbImages, dtype=np.bool)
                select[leave_out] = 0
                X = X[select,:]
                #remove leave_out from Call

                #do pca
                [eigenvalues, eigenvectors, mu] = pca(X,num_components=10)
                #visualize
                if show:
                    cv2.imshow('img',mu.reshape(400,400).astype(np.uint8))
                    cv2.waitKey(0) 
                #project leave_out on the subspace
                Yleave_out = project(eigenvectors, X[leave_out,:], mu )
        #        print Yleave_out
                #reconstruct leave_out
                Xleave_out2= reconstruct(eigenvectors, Yleave_out, mu)
                if show:
                    #show reconstructed image
                    Xleave_out2 = Xleave_out2*(256./(np.max(Xleave_out2)-np.min(Xleave_out2))) + np.min(Xleave_out2)
                    cv2.imshow('img',Xleave_out2.reshape(400,400).astype(np.uint8))
                    cv2.waitKey(0)        
                #classify leave_out
                bestError = float('inf')
                bestC = 0
                bestI = 0
                for i in range(nbImages-1):
                    y = project(eigenvectors, X[i,:], mu )
                    error = np.linalg.norm(y-Yleave_out)
                    if error < bestError:
                        bestError = error
                        bestC = C[i]
                        bestI = i
                print str(leave_out)+":"+str(bestC)+" - because of "+str(bestI)+" with error:"+str(bestError)
                if bestC == Call[leave_out]:
                    nbCorrect+=1
            
            #Print final result
            print str(nbCorrect)+"/"+str(nbImages)
    
