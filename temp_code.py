from Model import Model    
import time
m1 = Model()
m1._get_patients(14)

start_time = time.time()
m1._weight_matrix(m1.Patients)
print("--- %s seconds ---" % (time.time() - start_time))


           sum(m1.weight_matrix_ * m1.Patients[0].Teeth[:,0])

           img = plt.imread('C:/Users/tangc/Documents/ComVi/_Data/Radiographs/01.tif')
           fig = plt.figure()
           plt.imshow(img)
           plt.plot(m1.Patients[0].Teeth[:,0],m1.Patients[0].Teeth[:,1],'g.',markersize=1.5)