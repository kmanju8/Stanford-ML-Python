import numpy as np
from sigmoid import sigmoid

def validation(theta,X,Y,category):

    totalcorrect=0
    Y=Y.flatten()
    prediction=np.zeros(len(Y))

    mistakes={}
    
    for i in range(len(Y)):
        hypothesis=(category[np.argmax(np.matmul(X[i,:],theta.T))])
        if hypothesis==Y[i]:
            totalcorrect+=1
        else:
            if Y[i] in mistakes:
                mistakes[Y[i]]+=1
            else:
                mistakes[Y[i]]=1
    
    return(totalcorrect/len(Y),mistakes)
