from Model import Model    
import time
m1 = Model()
m1._get_patients(14)

start_time = time.time()
m1._procrustes_analysis(m1.Patients)
print("--- %s seconds ---" % (time.time() - start_time))


m1 = Model()
m1._get_patients(14)
shape_vectors = np.array([m1.Patients[i].Teeth for i in range(len(m1.Patients))])
mean = np.mean(shape_vectors, axis=0)
mean = np.reshape(mean, (-1,2))
min_x = min(mean[:,0])
min_y = min(mean[:,1])
mean[:,0] = [x - min_x for x in mean[:,0]]
mean[:,1] = [y - min_y for y in mean[:,1]]
mean = mean.flatten()



           sum(m1.weight_matrix_ * m1.Patients[0].Teeth[:,0])

           img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/01.tif')
           fig = plt.figure()
           plt.imshow(img)
           plt.plot(m1.Patients[0].Teeth[:,0],m1.Patients[0].Teeth[:,1],'g*',markersize=1.5)
           plt.plot(m1.Patients[1].Teeth[:,0],m1.Patients[1].Teeth[:,1],'r-',markersize=2)
           plt.plot(m1.Patients[10].Teeth[:,0],m1.Patients[10].Teeth[:,1],'b.',markersize=1.5)
           plt.show()