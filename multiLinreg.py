import numpy as np

def MLS(timeSeries, predictorsNum, miniTrainingSize):
    X= [ [0] *(1+ predictorsNum)  for _ in range(miniTrainingSize) ]
    # X is a miniTS by (1+pred) size matrix
    X_diff = np.ones( (miniTrainingSize, 1+predictorsNum) )
    Y = []
    # print( len(timeSeries))
    for i in range(0,miniTrainingSize):
        X[i] = timeSeries[  len(timeSeries)-(i)-(predictorsNum)-1 : len(timeSeries)-(i) ]
    
    # print(X)

    for i in range(miniTrainingSize):
        Y.append(X[i][0])
        X[i][0] = 1
    
    for i in range(0,miniTrainingSize):
        for j in range(2,predictorsNum):
            X_diff[i][j]=X[i][j]-X[i][j-1]
    X = np.array(X)
 

    beta = ( np.linalg.pinv(  (X_diff.transpose()).dot(X_diff)  ) ).dot( (X_diff.transpose()).dot(Y) )

    return beta
    
    



