import numpy as np

def analyze(X,Y):

    pseudo=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X))
    theta=np.matmul(pseudo,Y)

    return(theta)
