import numpy as np
from sigmoid import sigmoid

def cost_func(theta,X,Y):

    m=len(Y)
    totalcost=0

    hypothesis=sigmoid(np.matmul(X,theta))

    for i in range(m):
        #tidy up gradually to get proper vector notation

        if Y[i]==0:
            totalcost-=np.log(1-hypothesis[i])
        else:
            totalcost-=np.log(hypothesis[i])
    
    totalcost=totalcost/m
    
    return(totalcost)
    
def cost_grad(theta,X,Y):

    m=len(Y)
    totalgrad=np.zeros(len(X[0]))
    hypothesis=sigmoid(np.matmul(X,theta))
    
    for i in range(m):

        totalgrad+=(hypothesis[i]-Y[i])*X[i]

    return(totalgrad/m)
