import numpy as np
import pandas as pd
import scipy.io
from displayData import plotsamples
from oneVsAll import trainmulti
from predictOneVsAll import validation

#completed 1.4. Actually might overfit, hard to tell

#first need to load datafile into python from MATLAB database. Interesting project may to be to collect data set myself, clean, etc.
mat = scipy.io.loadmat('ex3data1.mat')
X = np.array(mat['X'])
y = np.array(mat['y'])
lr=0.001


#code to plot random selection of 100 digits, for visualization.
#X=(X/np.max(X))*255
#plotsamples(X)


#train model for each class
#labels are stored in y
#initial theta in oneVsAll


#THETA,category = trainmulti(X,y,lr)
'''
with open('solution.npy', 'wb') as f:
    np.save(f, THETA)
    np.save(f,category)

'''
with open('solution.npy', 'rb') as f:
    THETA = np.load(f)
    category = np.load(f)


perc,mistakes=validation(THETA,X,y,category)
print(f"{perc} of samples are correct")

#dictionary showing number of mistakes per digit. This displays ACTUAL values.
print(mistakes)


