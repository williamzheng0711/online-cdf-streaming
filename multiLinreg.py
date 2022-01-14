# This file is just for using the normal equation (ESTR1004) to do the multiple linear regression.

import numpy as np

def MultipleLinearRegression(timeSeries, predictorsNum, miniTrainingSize):
    X = np.zeros( (miniTrainingSize,predictorsNum) )
    Y = np.zeros( (miniTrainingSize,1) )
    # print(len(timeSeries))
    for i in range(miniTrainingSize):
        X[i] = timeSeries[  len(timeSeries)-i-(predictorsNum) : len(timeSeries)-i ][::-1]
        
        # print(X[i])
        Y[i] = X[i][0]
    
    for i in range(miniTrainingSize):
        X[i][0] = 1
        
    beta = ( np.linalg.pinv(  (X.transpose()).dot(X)  ) ).dot( (X.transpose()).dot(Y) )
    return beta
    
    



