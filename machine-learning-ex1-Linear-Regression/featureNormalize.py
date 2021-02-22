import numpy as np

def xnorm(X):

    means=np.mean(X,axis=0)
    stdevs=np.std(X,axis=0)

    

    for i in range(1,len(X[0])):
        #normalize each row
        X[:,i]=X[:,i]-means[i]
        X[:,i]=X[:,i]/means[i]

    
    return(X)
