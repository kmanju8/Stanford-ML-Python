import csv
import numpy as np
import matplotlib.pyplot as plt
from computeCostMulti import cost_eval
from gradientDescentMulti import graddesc
from featureNormalize import xnorm
from mpl_toolkits import mplot3d
from normalEqn import analyze

def f(x,y,theta):

    ans =  theta[0]+theta[1,]*x+theta[2]*x
    
    return ans

#Currently on exercise 3.3, last one.

#reading data from file
#data is size in sq feet, bedrooms, value

dataset = np.genfromtxt('ex1data2.txt', delimiter =",")
#adds row of ones onto X as well




X = np.hstack((np.ones((len(dataset),1)),dataset[:,:-1]))
Y = dataset[:,-1]


theta=np.zeros((len(X[0]),1))



X=xnorm(X)

feet=dataset[:,0]
rooms=dataset[:,1]





#plt.plot(population,profit,'x')
#plt.ylabel("Profit in $10,000s")
#plt.xlabel("Population of City in 10,000s")
#plt.show()


#-----------------Setting up for gradient descent----------------------

#need transpose technically, but for now this is good for simplicity.
'''X=np.vstack((np.ones(len(dataset)),population))
theta=np.zeros((2,1))'''


iterations=1500
alpha = 0.1

#------------------Area to run external functions-----------------------


'''#plot new line for 5 to 25
xreg = np.linspace(5,25,100)
yreg = xreg*theta[1,0]+theta[0,0]
plt.plot(xreg,yreg)

plt.show()'''
theta=graddesc(X,Y,theta,iterations,alpha)

'''#plotting in 3d as a test
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1],X[:,2],Y)

#plot linear fit
x = np.linspace(-0.5, 0.5, 30)
y = np.linspace(-0.5, 0.5, 30)
A,B=np.meshgrid(x,y)
z = f(A,B,theta)
ax.contour3D(A,B,z)

plt.show()'''

print(theta)

#theta=analyze(X,Y)
print("Analytical solution:", theta)


#plotting in 3d as a test
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1],X[:,2],Y)

#plot linear fit
x = np.linspace(-0.5, 0.5, 30)
y = np.linspace(-0.5, 0.5, 30)
A,B=np.meshgrid(x,y)
z = f(A,B,theta)
ax.contour3D(A,B,z)

plt.show()
