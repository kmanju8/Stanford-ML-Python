import numpy as np
import pandas as pd
import scipy.optimize
from lrCostFunction import cost_func, cost_grad

def trainmulti(X,Y,lr):

    #group data. Find how many classes exist
    classes={}
    for i in Y[:,0]:
        if i not in classes:
            classes[i]=0


    THETA=np.zeros((len(classes),len(X[0])))

    category=[]
    #optimizing theta for each classifier
    for digit in classes:

        print(digit)
        
        #will need to reinitialize Y each time to have only 1s and 0s.
        Yspec = np.zeros(Y.shape)
        for i in range(len(Y[:,0])):
            if Y[i,0]==digit:
                Yspec[i,0]=1
        
        optimization=scipy.optimize.minimize(cost_func,THETA[len(category),:],(X,Yspec,lr), jac=cost_grad)
        THETA[len(category),:]=optimization.x
        
        #to tell which row of theta is for which category
        category=category+[digit]
        print(THETA)


    return(THETA,category)
