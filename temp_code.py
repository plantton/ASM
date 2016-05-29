          idx = []
          for j in range(dataframe.index.shape[0]):        
                if int(dataframe.index[j][9:-4].split('-')[0]) == i:
                   idx.append(dataframe.index[j])
          return dataframe.loc[idx, :] 
        
        
for j, str in enumerate(ldlist):
    if j < 5:
     print j
     print str
     

