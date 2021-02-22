import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
from sigmoid import sigmoid
from costFunction import cost_func,cost_grad


def validation(theta,X,Y):

    
    hypothesis=sigmoid(np.transpose(np.matmul(X,np.transpose(theta))))

    predictions=np.round(hypothesis)

    total=len(predictions)
    correct=0

    for i in range(total):
        if predictions[i]==Y[i]:
            correct+=1

    return(correct/total)
    

    

#currently on 1.2.4

#data initialization
dataset = np.genfromtxt('ex2data1.txt', delimiter =",")

x=dataset[:,0]
y=dataset[:,1]
X=np.hstack((np.ones((len(dataset),1)),dataset[:,:-1]))
labels=dataset[:,2]



#initial theta

theta = np.zeros((3,1))

print(cost_func(theta,X,labels))
theta=scipy.optimize.minimize(cost_func,theta,(X,labels), jac=cost_grad)

print(theta)
print(cost_func(theta.x,X,labels))




df = pd.DataFrame(dict(x=x,y=y, label=labels))

groups = df.groupby('label')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)

ax.legend()

#plot decision boundary

x = np.linspace(20, 120, 1000)
y = (theta.x[0]+theta.x[1]*x)/(-theta.x[2])
ax.plot(x,y)


#validation?
accuracy=validation(theta.x,X,labels)
print(accuracy*100,"% of predictions are correct")

plt.show()


