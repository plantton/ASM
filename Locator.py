# Class Align Teeth image
import cv2
import cv2.cv as cv
import os
import numpy as np
import time
import fnmatch


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
            
        def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	      for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])    

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
    
