  
     def get_X():
         pts = [4,3,2,1]
         w = range(1,5,1)
         return sum([w[i]*pts[i] for i in range(len(pts))])
