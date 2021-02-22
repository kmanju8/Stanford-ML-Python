import numpy as np

#finds value for cost function J(theta)
def cost_eval(X,Y,theta):

    m=len(Y)
    totalcost=0

    hypothesis=np.matmul(X,theta)

    for i in range(m):
        #tidy up graduatlly to get proper vector notation
        
        totalcost+=(hypothesis[i]-Y[i])**2
    
    totalcost=totalcost/(2*m)
    
    return totalcost
    
