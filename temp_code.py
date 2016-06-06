from Model import Model    
import time
m1 = Model()
m1._get_patients(14)

start_time = time.time()
m1._procrustes_analysis(m1.Patients)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
[evals,evecs,num] = m1._PCA(m1.Patients)
print("--- %s seconds ---" % (time.time() - start_time))



m1 = Model()
m1._get_patients(14)
start_time = time.time()
m1._procrustes_analysis(m1.Patients)
print("--- %s seconds ---" % (time.time() - start_time))
mean_shape = m1._get_mean_shape(m1.Patients)
[evals,evecs,num] = m1._PCA(m1.Patients)
tot = sum(evals)
ratio_s = evals / tot
test = np.dot(evecs,_all_teeths.T)
plot(test[:,0])

m1._weight_matrix(m1.Patients)
[a,b]=m1.Patients[1]._alignment_parameters(m1.Patients[2],m1.weight_matrix_)


shape_vectors = np.array([np. ravel(m1.Patients[i].Teeth) for i in range(len(m1.Patients))])
mean = np.mean(shape_vectors, axis=0)
mean = np.reshape(mean, (-1,2))
min_x = min(mean[:,0])
min_y = min(mean[:,1])
mean[:,0] = [x - min_x for x in mean[:,0]]
mean[:,1] = [y - min_y for y in mean[:,1]]
mean = mean.flatten()


m1.Patients[1:] = [s.align_to_shape(m1.Patients[0], m1.weight_matrix_) for s in m1.Patients[1:]]


img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/01.tif')
fig = plt.figure()
plt.imshow(img)
for i in m1.Patients:
       plot(i.Teeth[:,0],i.Teeth[:,1],'.',markersize=1.5)
plt.show()

           sum(m1.weight_matrix_ * m1.Patients[0].Teeth[:,0])

           img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/01.tif')
           fig = plt.figure()
           plt.imshow(img)
           plt.plot(m1.Patients[9].Teeth[:,0],m1.Patients[0].Teeth[:,1],'g*',markersize=1.5)
           plt.plot(m1.Patients[1].Teeth[:,0],m1.Patients[1].Teeth[:,1],'r.',markersize=2)
           plt.plot(m1.Patients[7].Teeth[:,0],m1.Patients[10].Teeth[:,1],'b.',markersize=1.5)
           plt.show()
           
               num_patients = len(m1.Patients)   
               _all_teeths = np.zeros(shape=(num_patients,6400))
               for i,t in enumerate(m1.Patients):
                   _all_teeths[i,:] = np.ravel(t.Teeth)
               # Use inalg.eig to do eigenvalue decomposition. 
               # Inspired from https://github.com/andrewrch/active_shape_models/blob/master/active_shape_models.py
               cov = np.cov(_all_teeths, rowvar=0)
               evals, evecs = np.linalg.eig(cov)
               evals = evals.real
               evecs = evecs.real
               ratio = np.divide(evals,sum(evals))
               _evals = evals[:len(ratio[np.cumsum(ratio)<0.99])]
               _evecs = evecs[:len(_evals)]
               return (_evals, _evecs,len(_evals)) 