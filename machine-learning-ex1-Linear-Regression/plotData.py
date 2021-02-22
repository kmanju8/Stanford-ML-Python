import csv
import numpy as np
import matplotlib.pyplot as plt
from computeCost import cost_eval
from gradientDescent import graddesc

#reading data from file
dataset = np.genfromtxt('ex1data1.txt', delimiter =",")
population = np.array(dataset[:,0])
profit = np.array(dataset[:,1])


#plt.plot(population,profit,'x')
#plt.ylabel("Profit in $10,000s")
#plt.xlabel("Population of City in 10,000s")
#plt.show()


#-----------------Setting up for gradient descent----------------------

#need transpose technically, but for now this is good for simplicity.
X=np.vstack((np.ones(len(dataset)),population))
theta=np.zeros((2,1))

iterations=1500
alpha = 0.01

#------------------Area to run external functions-----------------------

theta=graddesc(X,profit,theta,iterations,alpha)

plt.plot(population,profit,'x')
plt.ylabel("Profit in $10,000s")
plt.xlabel("Population of City in 10,000s")

#plot new line for 5 to 25
xreg = np.linspace(5,25,100)
yreg = xreg*theta[1,0]+theta[0,0]
plt.plot(xreg,yreg)

plt.show()


print(theta)
