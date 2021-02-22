import numpy as np
import matplotlib.pyplot as plt


def graddesc(X,Y,theta,iterations,alpha):

    m=len(Y)
    

    for i in range(iterations):

        '''if i%200==0:
            xreg = np.linspace(5,25,100)
            yreg = xreg*theta[1,0]+theta[0,0]
            plt.plot(xreg,yreg,label="a")'''

        delx0=0
        delx1=0
        
        for j in range(m):
        
            delx0+=(np.matmul(X[:,j],theta)-Y[j])
            delx1+=(np.matmul(X[:,j],theta)-Y[j])*X[1,j]

        theta[0,0]-=(alpha*delx0)/m
        theta[1,0]-=(alpha*delx1)/m

    return(theta)
