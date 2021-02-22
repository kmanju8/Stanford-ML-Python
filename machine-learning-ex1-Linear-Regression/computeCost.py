import numpy as np

#finds value for cost function J(theta)
def cost_eval(X,Y,theta):

    m=len(Y)
    totalcost=0

    for i in range(m):
        #tidy up graduatlly to get proper vector notation
        
        totalcost+=(np.matmul(X[:,i],theta)-Y[i])**2
    
    totalcost=totalcost/(2*m)
    
    return totalcost
    
