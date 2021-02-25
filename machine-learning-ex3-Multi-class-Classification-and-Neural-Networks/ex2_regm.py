import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
from sigmoid import sigmoid
from lrCostFunction import cost_func, cost_grad
from mapFeature import polymap


def validation(theta,X,Y):
    
    hypothesis=sigmoid(np.transpose(np.matmul(X,np.transpose(theta))))
    predictions=np.round(hypothesis)
    total=len(X[0,:])
    correct=0

    for i in range(total):
        if predictions[i]==Y[i]:
            correct+=1

    return(correct/total)
#--------------------------------------------------------------------------
    
#finished

#data initialization
dataset = np.genfromtxt('ex2data2.txt', delimiter =",")

x=dataset[:,0]
y=dataset[:,1]
X=np.hstack((np.ones((len(dataset),1)),dataset[:,:-1]))
labels=np.array([dataset[:,2]]).T
#learning rate
lr=0.000001

X=polymap(x,y)




#initial theta
theta = np.zeros((np.size(X[0]),1))


optimization=scipy.optimize.minimize(cost_func,theta,(X,labels,lr), jac=cost_grad)

theta=optimization.x





#--------------------------------------------------------
df = pd.DataFrame(dict(x=x,y=y, label=dataset[:,2]))

groups = df.groupby('label')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)

ax.legend()

#plot decision boundary

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u),len(v)))

for i in range(len(u)):
    for j in range(len(v)):
        z[i,j]=np.dot(polymap(u[i],v[j]),theta)
z=np.transpose(z)

#just need to plot contour of 0 now
ax.contour(u,v,z,[-0.1,0.1])

#--------------------------------------------------
#validation?
accuracy=validation(theta,X,labels)
print(accuracy*100,"% of predictions are correct")

plt.show()


