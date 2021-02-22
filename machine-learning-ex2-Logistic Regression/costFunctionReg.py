import numpy as np
from sigmoid import sigmoid

#have to modify for added regression terms
def cost_func(theta,X,Y,learningrate):

    m=len(Y)
    totalcost=0


    hypothesis=sigmoid(np.matmul(X,theta))
    regbonus=0


    for i in range(m):
        #tidy up gradually to get proper vector notation

        if Y[i]==0:
            totalcost-=np.log(1-hypothesis[i])
        else:
            totalcost-=np.log(hypothesis[i])
    
    for i in range(1,len(theta)):
        regbonus+=theta[i]**2

    totalcost+=regbonus*(learningrate/2)
    
    totalcost=totalcost/m
    
    return(totalcost)
    
def cost_grad(theta,X,Y,learningrate):

    m=len(Y)
    totalgrad=np.zeros(len(X[0]))
    hypothesis=sigmoid(np.matmul(X,theta))
    
    for i in range(m):
        totalgrad+=(hypothesis[i]-Y[i])*X[i]

    for i in range(1,len(totalgrad)):
        totalgrad[i]+=(learningrate/m)*theta[i]

    return(totalgrad/m)
