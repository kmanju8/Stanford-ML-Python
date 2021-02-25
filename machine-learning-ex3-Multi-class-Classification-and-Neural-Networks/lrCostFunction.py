import numpy as np
from sigmoid import sigmoid


def cost_func(theta,X,Y,learningrate):

    m=len(Y)
    theta=theta.reshape(-1,1)
    
    hypothesis=sigmoid(np.matmul(X,theta))
    totalcost=(np.matmul(-Y.T,np.log(hypothesis))-np.matmul((np.ones(np.shape(Y))-Y).T,(np.log(1-hypothesis))))/m

    #regularization
    totalcost+=(learningrate/2*m)*(np.matmul(theta.T,theta)-(theta[0])**2)
     
    return(totalcost)
  
def cost_grad(theta,X,Y,learningrate):

    m=len(Y)
    theta=theta.reshape(-1,1)

    hypothesis=sigmoid(np.matmul(X,theta))
    totalgrad=np.matmul(X.T,(hypothesis-Y))/m
    totalgrad[1:,0]+=((learningrate/m)*theta[1:,0])


    return(totalgrad.flatten())
