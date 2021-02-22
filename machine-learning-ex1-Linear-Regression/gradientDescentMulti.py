import numpy as np

def graddesc(X,Y,theta,iterations,alpha):

    m=len(Y)
    Y=Y.reshape(-1,1)
    
    for i in range(iterations):

        if i%200==0:
            print(i, theta)
            
            #xreg = np.linspace(5,25,100)
            #yreg = xreg*theta[1,0]+theta[0,0]
            #plt.plot(xreg,yreg,label="a")
        #make array of delxs, probably numpy array
        delx=np.zeros((len(theta),1))

        error=(np.matmul(X,theta)-Y)


        #probs a lin alg rewrite
        for k in range(len(X[0])):
            delx[k,0]+=np.matmul(X[:,k],error)
            

        
        theta-=(alpha*delx)/m
        

    return(theta)
