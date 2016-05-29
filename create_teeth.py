class Teeth:
# A class represents 8 teeth for a certain patient.

      lddir = ASMdir+'/_Data/Landmarks/original/'

# range of i is between 1 to 8.
      def create_teeth(i):
          idx = []
          for j in range(dataframe.index.shape[0]):        
                if int(dataframe.index[j][9:-4].split('-')[0]) == i:
                   idx.append(dataframe.index[j])
          return dataframe.loc[idx, :] 
        
        
