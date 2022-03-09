# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Mar. 9, 2022

from cmath import e
import csv
from math import exp, floor, pi, sin, sqrt
from multiprocessing.context import ForkProcess
from operator import index, indexOf, ne
import numpy as np
from numpy.core.numeric import False_
from scipy.stats import laplace
from scipy.stats import laplace_asymmetric
from scipy.stats.morestats import boxcox_normmax
from numpy.core.fromnumeric import argmax, mean, size, var
from numpy.lib.function_base import append, kaiser
import utils as utils
from torch.autograd import Variable
import time as tm
import pandas as pd
import matplotlib.pyplot as pyplot
from statistics import NormalDist
from numpy import linalg as LA, quantile
from scipy.stats import norm, gaussian_kde
from sklearn.neighbors import KernelDensity



B_IN_MB = 1024*1024

whichVideo = 5
# Note that FPS >= 1/networkSamplingInterval
FPS = 10

# Testing Set Size
howLongIsVideoInSeconds = 300

# Training Data Size
timePacketsDataLoad = 4000000

network_trace_dir = './dataset/fyp_lab/'

networkEnvTime = []
networkEnvPacket= []
count = 0
initialTime = 0

# load the mock data from our local dataset
for suffixNum in range(whichVideo,whichVideo+1):
    with open( network_trace_dir+ str(suffixNum) + ".txt" ) as traceDateFile:
        for eachLine in traceDateFile:
            parse = eachLine.split()
            if (count==0):
                initialTime = float(parse[0])
            nowFileTime = float(parse[0]) 
            networkEnvTime.append(nowFileTime - initialTime)
            networkEnvPacket.append( float(parse[1]) / B_IN_MB ) 
            count = count  +1 

# All things above are  "environment"'s initialization, 
# which cannot be manipulated by our algorithm.
############################################################################
# All things below are of our business


ratioTrain = 0.5

trainingDataLen =  floor(ratioTrain * len(networkEnvPacket))

timeTrack = 0
amount = 0
sampleThroughputRecord = []
for numberA in range(0,trainingDataLen):
    amount = amount + networkEnvPacket[numberA]
    if ( ( networkEnvTime[numberA] - timeTrack ) > 1 / FPS ):
        throughputLast = amount / ( networkEnvTime[numberA] - timeTrack  )
        timeTrack = networkEnvTime[numberA]
        sampleThroughputRecord.append( throughputLast )
        amount = 0


pGamma = 0.2
pEpsilon = 0.1

testingTimeStart = timeTrack


def uploadProcess(user_id, minimal_framesize, estimatingType, pLogCi, forTrain, pTrackUsed, pForgetList):
    frame_prepared_time = []
    throughputHistoryLog = pLogCi
    realVideoFrameSize = []
 
    # This is to SKIP the training part of the data.
    # Hence ensures that training data is not over-lapping with testing data
    tempTimeTrack = testingTimeStart
    runningTime = testingTimeStart
    for _ in range( howLongIsVideoInSeconds * FPS ):
        frame_prepared_time.append(tempTimeTrack)
        tempTimeTrack = tempTimeTrack + 1/FPS
    
    uploadDuration = 0

    # initialize the C_0 hat (in MB), which is a "default guess"
    throughputEstimate = (1/FPS) * mean(sampleThroughputRecord) 
    count_skip = 0

    # Note that the frame_prepared_time every time is NON-SKIPPABLE
    for singleFrame in range( howLongIsVideoInSeconds * FPS ):
        # if (singleFrame % 1000 ==0): 
        #     print(singleFrame)
        ########################################################################################

        if (runningTime - testingTimeStart > howLongIsVideoInSeconds):
            break 

        # To determine if singleFrame is skipped
        if ( singleFrame < howLongIsVideoInSeconds * FPS - 1 and runningTime > frame_prepared_time[singleFrame + 1] ):
            count_skip = count_skip + 1
            continue

        if (singleFrame >0 and ( runningTime < frame_prepared_time[singleFrame])): 
            # 上一次的傳輸太快了，導致新的幀還沒生成出來
            # Then we need to wait until singleframe is generated and available to send.
            runningTime = frame_prepared_time[singleFrame]
        
        if (runningTime - frame_prepared_time[singleFrame] > 1/FPS):
            count_skip = count_skip + 1
            continue

        #######################################################################################
        # Anyway, from now on, the uploader is ready to send singleFrame
        # In this part, we determine what is the suggestedFrameSize 
        # We initialize the guess as minimal_framesize
        suggestedFrameSize = -np.Infinity

        delta = runningTime -  frame_prepared_time[singleFrame]
        T_i = (1/FPS - delta)
        r_i = T_i * FPS
        
        if (estimatingType == "ProbabilityPredict" and len(throughputHistoryLog) > 0 ):
            # print(len(throughputHistoryLog))
            # print("here!!!")
            # throughputHistoryLog = throughputHistoryLog[ len(throughputHistoryLog) -1 - lenLimit : len(throughputHistoryLog)-1]
            C_iMinus1=throughputHistoryLog[-1]
            subLongSeq = [throughputHistoryLog[i+1] for _, i in zip(throughputHistoryLog,range(len(throughputHistoryLog))) 
                if ( (abs((throughputHistoryLog[i]-C_iMinus1)/C_iMinus1)< 0.05) and i<len(throughputHistoryLog)-1 ) ]
            try: 
                tempCihat = quantile(subLongSeq, pEpsilon)
                throughputEstimate = ( 1 + pGamma/r_i ) * tempCihat
                suggestedFrameSize = throughputEstimate * T_i
                # pyplot.hist(subLongSeq, bins=80)
                # pyplot.show()
            except:
                suggestedFrameSize = T_i * mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed) : len(throughputHistoryLog)])
     
        elif (estimatingType == "A.M." and len(throughputHistoryLog) > 0 ):
            suggestedFrameSize = T_i * mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ])

        # need to judge in my new rule
        # if suggested value < minimal size, then we discard the frame
        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize )

        # Until now, the suggestedFrameSize is fixed.
        #######################################################################################

        # The following function will calculate t_f.
        uploadFinishTime = utils.packet_level_frame_upload_finish_time( runningTime= runningTime,
                                                                        packet_level_data= networkEnvPacket,
                                                                        packet_level_timestamp= networkEnvTime,
                                                                        framesize= thisFrameSize)
        
        # We record the sent frames' information in this array.
        realVideoFrameSize.append(thisFrameSize)

        uploadDuration = uploadFinishTime - runningTime
        # To update the current time clock, now we finished the old frame's transmission.
        runningTime = runningTime + uploadDuration 

        # Here we calculated the C_{i-1}
        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)

        # if (len(throughputHistoryLog)>0 and estimatingType == "ProbabilityPredict"):
        #     if (len(print(len(throughputHistoryLog))) > lenLimit ):
        #         throughputHistoryLog = throughputHistoryLog[1:]
    


    return [ sum(realVideoFrameSize), [], count_skip, minimal_framesize, len(realVideoFrameSize)]



number = 100

mAxis = [5,16,128]
xAxis =  np.linspace(0.0000001, 0.2 ,num=number, endpoint=True)


lenLimit = 600*FPS
bigHistorySequence = sampleThroughputRecord[0:600*FPS]
print(len(bigHistorySequence))


toPlot = 0

for trackUsed in mAxis:
    y1Axis = []
    y2Axis = []
    y3Axis = []
    z1Axis = []
    z2Axis = []

    for x in xAxis:
        a = uploadProcess('dummyUsername1', x , "A.M.", pLogCi=bigHistorySequence, forTrain=False, pTrackUsed=trackUsed, pForgetList=[])
        b = uploadProcess('dummyUsername2', x , "ProbabilityPredict", pLogCi=bigHistorySequence , forTrain=False, pTrackUsed=trackUsed, pForgetList=[])
        count_skipA = a[2]
        count_skipB = b[2]
        y1Axis.append(count_skipA/(howLongIsVideoInSeconds*FPS))
        y2Axis.append(count_skipB/(howLongIsVideoInSeconds*FPS))

        z1Axis.append(a[0])
        z2Axis.append(b[0])

        print("A.M.(M="+str(trackUsed)+"): " +  str(a[0]) + " " + str(count_skipA/(howLongIsVideoInSeconds*FPS)) + " with min-size: " + str(a[3]) )
        print("OurMethod: " + str(b[0]) + " " + str(count_skipB/(howLongIsVideoInSeconds*FPS)))


    # print("Mean of this network: " + str(mean(networkEnvTP)))
    # print("Var of ~: " + str(var(networkEnvTP)))


    toPlot += 1
    pyplot.subplot( len(mAxis),2,toPlot)
    pyplot.xlabel("Minimal Each Frame Size (in MB)")
    pyplot.ylabel("Loss Rate")
    pyplot.plot(xAxis, y2Axis, '-s', color='blue', markersize=1, linewidth=1)
    pyplot.plot(xAxis, y1Axis, '-s', color='red', markersize=1, linewidth=1)
    pyplot.legend( ["Empirical Condt'l", "A.M. M=" + str(trackUsed),], loc="best")

    toPlot += 1
    pyplot.subplot( len(mAxis),2,toPlot)
    pyplot.xlabel("Minimal Each Frame Size (in MB)")
    pyplot.ylabel("Data sent in" + str(howLongIsVideoInSeconds) +" sec" )
    pyplot.plot(xAxis, z2Axis, '-s', color='blue',
                markersize=1, linewidth=1)
    pyplot.plot(xAxis, z1Axis, '-s', color='red',
                markersize=1, linewidth=1)
    pyplot.legend( ["Empirical Condt'l", "A.M. M=" + str(trackUsed),], loc="best")

pyplot.show()