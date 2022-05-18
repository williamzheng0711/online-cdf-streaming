from math import floor
from time import time
from unittest import skip
from lightgbm import train
import numpy as np
from numpy.core.fromnumeric import mean
import utils as utils
import matplotlib.pyplot as pyplot
from numpy import cumsum, quantile
import pandas as pd
from darts import TimeSeries
import numpy as np
from darts.models import ExponentialSmoothing
from darts.models import ARIMA


network_trace_dir = './dataset/network_trace/1h_less/'
howmany_Bs_IN_1Mb = 1024*1024/8


whichVideo = 12
# Testing Set Size
timeInterval = 0.04


networkEnvTime = []
networkEnvTrace= []
count = 0
timeMark = 0

for suffixNum in range(whichVideo,whichVideo+1):
    with open( network_trace_dir+ str(suffixNum) + ".csv" ) as traceDateFile:
        for eachLine in traceDateFile:
            parse = eachLine.split()
            networkEnvTrace.append( float(parse[0]) / howmany_Bs_IN_1Mb ) 
            networkEnvTime.append(timeMark)
            count = count  +1 
            timeMark += timeInterval



def runningAlgos(trainingTimeLen, testingTimeLen):

    ARIMA_Result = []
    Condl_Prob_Result = []
    ExponentialSmoothing_Result = []

    trainingSlotsLen = int(trainingTimeLen/timeInterval)
    testingSlotsLen = int(testingTimeLen/timeInterval)
    trueThroughput = networkEnvTrace[trainingSlotsLen:trainingSlotsLen+testingSlotsLen]


    for j in range(testingSlotsLen):
        if (j%25==0): print(str(j/testingSlotsLen))
        series = TimeSeries.from_values( np.array( networkEnvTrace[0:trainingSlotsLen+testingSlotsLen] ) )
        train, val = series[0:trainingSlotsLen+j], series[trainingSlotsLen+j:trainingSlotsLen+testingSlotsLen]

        exp_model = ExponentialSmoothing()
        exp_model.fit(train)

        arima_model = ARIMA(p = 4, d = 2, q= 1)
        arima_model.fit(train)

        exp_predict =  exp_model.predict(1, num_samples=20)
        arima_predict = arima_model.predict(1, num_samples=20)

        ExponentialSmoothing_Result.append(exp_predict)
        ARIMA_Result.append(arima_predict)
    
    print(ARIMA_Result)

    # pyplot.plot(trueThroughput)
    # pyplot.plot(ARIMA_Result)
    # pyplot.plot(np.array(ExponentialSmoothing_Result))
    # pyplot.show()

runningAlgos(120,2)



