import numpy as np

'''% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%'''

#will leave till later on in the assignment

#X1 and X2 will be the different variables, so each one column of X.

def polymap(X1,X2):
    degree = 6

    length=int(((degree+2)*(degree+1))/2)

    newfeatures=np.ones((np.size(X1),length))

    index=1
    for i in range(1,degree+1):
        
        for j in range(i+1):
            newfeatures[:,index]=(X1**(i-j))*(X2**j)
            index+=1

    return(newfeatures)
            
            

