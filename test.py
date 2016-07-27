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
        
        image = cv2.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/03.tif')
        loc = t11._obj_dec_(image,40,400,400)
        clone = image.copy()
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.circle(clone, (loc[0],loc[1]),50,255,-1)
        small = cv2.resize(clone, (0,0), fx=0.3, fy=0.3)
        #plt.imshow(clone)
        #plt.Circle((1407, 1172),500,color='r')
        #plt.show()
        
        cv2.imshow("Window", small)
        
        
        
        def auto_canny(image, sigma=0.20):
           	# compute the median of the single channel pixel intensities
           	v = np.median(image)
            
           	# apply automatic Canny edge detection using the computed median
           	lower = int(max(0, (1.0 - sigma) * v))
           	upper = int(min(255, (1.0 + sigma) * v))
           	edged = cv2.Canny(image, lower, upper)
            
           	# return the edged image
           	return edged
        
        # test image reader
        #file_in = 'C:/Users/tangc/Documents/ComVi/ASM/eigenteeth2/03.jpg'
        file_in = 'C:/Users/tangc/Documents/ComVi/_Data/Radiographs/03.tif'
        img = cv2.imread(file_in)
        #cv2.imshow("ORIGINAL", img)
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray[loc[]]
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(25,25))
        cl1=clahe.apply(gray)
        
        blur = cv2.GaussianBlur(cl1,(3,3),0)
        blur = cv2.bilateralFilter(blur, 2, 20, 20)
        #blur = cv2.medianBlur(cl1, 13)
        
        
        #cv2.imshow("CLAHE", blur)
        #gray = cv2.equalizeHist(gray)
        
        #blur = cv2.GaussianBlur(cl1,(3,3),0)
        #blur = cv2.medianBlur(blur,3)
        #blur = cv2.fastNlMeansDenoising(blur,50,10,7,21)
        
        
        #wide = cv2.Canny(blur,10,200)
        #tight = cv2.Canny(blur,225,250)
        auto = auto_canny(blur)
        cv2.imshow("AUTO", auto)

        
        cv2.imshow("Original", img)
        cv2.imshow("Edges", np.hstack([wide, tight, auto]))
	cv2.waitKey(0)
        
        
        #blur = cv2.Canny(blur, 15, 150)
        
        # convolute with proper kernels
        #laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        #laplacian = cv2.convertScaleAbs(laplacian)
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
        
        #
        absX = cv2.convertScaleAbs(sobelx)
        absY = cv2.convertScaleAbs(sobely)
        dst = cv2.addWeighted( absX, 0.5, absY, 0.5,0)
        #cv2.imshow('img', grad)
        laplacian = cv2.Laplacian(dst, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)

        
        plt.subplot(2,2,1),plt.imshow(dst,cmap = 'gray')
        plt.title('Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(absX,cmap = 'gray')
        plt.title('absX'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(absY,cmap = 'gray')
        plt.title('absY'), plt.xticks([]), plt.yticks([])
        
        plt.show()
        
        
import numpy
    #a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
    numpy.savetxt("C:/Users/tangc/Documents/ComVi/ASM/random.txt", m1.Patients[0].Teeth, delimiter=",")
    #with open("C:/Users/tangc/Documents/ComVi/ASM/random.csv", "w") as fp:
    #    fp.write(data)
        
        cv2.imshow('Result', dst)
        grey_image = cv.CreateImage(cv.GetSize(grey), 8, 1)
        df_dx = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_16S, 1)
        if show:
            vis = img.copy()
            #cv2.rectangle(vis, (rects[0], rects[1]), (rects[2], rects[3]), (0, 255, 0), 2)
            cv2.imshow('img', vis)
            cv2.waitKey(0)
        