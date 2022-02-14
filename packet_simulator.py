# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Nov. 7, 2021

from cmath import e
import csv
from math import exp, floor, pi, sin, sqrt
from multiprocessing.context import ForkProcess
from operator import index, indexOf, ne
from shutil import which
import numpy as np
from numpy.core.numeric import False_
from scipy.stats import laplace
from scipy.stats import laplace_asymmetric
from scipy.stats.morestats import boxcox_normmax
from sklearn.preprocessing import MinMaxScaler
from numpy.core.fromnumeric import argmax, mean, size, var
from numpy.lib.function_base import append, kaiser
import utils as utils
from torch.autograd import Variable
import torch
import time as tm
import pandas as pd
import os
import matplotlib.pyplot as pyplot
from statistics import NormalDist
from numpy import linalg as LA



MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000.0*1000.0




whichVideo = 2
# Note that FPS >= 1/networkSamplingInterval
FPS = 25

# Testing Set Size
howLongIsVideoInSeconds = 180

# Training Data Size
timePacketsDataLoad = 300000

network_trace_dir = './dataset/fyp_lab/'

networkEnvTime = []
networkEnvPacket= []
count = 0
initialTime = 0

# load the mock data from our local dataset
for suffixNum in range(whichVideo,whichVideo+1):
    with open( network_trace_dir+ str(suffixNum) + ".txt" ) as file1:
        for line in file1:
            parse = line.split()
            if (count==0):
                initialTime = float(parse[0])
            nowTime = float(parse[0]) 
            networkEnvTime.append(nowTime - initialTime)
            networkEnvPacket.append( float(parse[1]) / B_IN_MB ) 
            count = count  +1 

# All things above are  "environment"'s initialization, 
# which cannot be manipulated by our algorithm.
############################################################################
# All things below are of our business

ratioTrain = 0.5
ratioTest = 1 - ratioTrain

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

# Until now, we know the empirical maximum.
#################

startPoint = np.quantile(sampleThroughputRecord, 0.005)
endPoint = np.quantile(sampleThroughputRecord, 0.995)
MIN_TP = min(sampleThroughputRecord)
MAX_TP = max(sampleThroughputRecord)

samplePoints = 70
marginalSample = 2
        
if (startPoint!=0):
    binsMe = np.concatenate(( np.linspace( MIN_TP,startPoint, marginalSample, endpoint=False) , 
                            np.linspace( startPoint, endPoint, samplePoints, endpoint=False) ,  
                            np.linspace( endPoint, MAX_TP, marginalSample, endpoint=True)  ), 
                            axis=0)
else:
    binsMe = np.concatenate(( np.linspace( startPoint, endPoint, samplePoints, endpoint=False) ,  
                              np.linspace( endPoint, MAX_TP, marginalSample, endpoint=True)  ), 
                              axis=0)

probability  = [ [0] * len(binsMe)  for _ in range(len(binsMe))]


# Until here the contingency table's size and bins are fixed.
#################


pGamma = 0.2
pEpsilon = 0.5

testingTimeStart = timeTrack


def uploadProcess(user_id, minimal_framesize, estimatingType, probability, forTrain, pTrackUsed, pForgetList):
    frame_prepared_time = []
    frame_start_send_time = []
    throughputHistoryLog = []
    realVideoFrameSize = []
    readVideoFrameNo = []
    probabilityModel = np.array(probability)
    toBeDeleted = pForgetList
 
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
        if ( singleFrame < howLongIsVideoInSeconds * FPS - 1 and runningTime > frame_prepared_time[singleFrame + 1] + pGamma/FPS):
            count_skip = count_skip + 1
            continue

        if (singleFrame >0 and ( runningTime <= frame_prepared_time[singleFrame])): 
            # 上一次的傳輸太快了，導致新的幀還沒生成出來
            # Then we need to wait until singleframe is generated and available to send.
            runningTime = frame_prepared_time[singleFrame]

        #######################################################################################
        # Anyway, from now on, the uploader is ready to send singleFrame
        # In this part, we determine what is the suggestedFrameSize 
        # We initialize the guess as minimal_framesize
        suggestedFrameSize = minimal_framesize

        delta = runningTime -  frame_prepared_time[singleFrame]
        T_i = (1/FPS - delta)
        r_i = T_i * FPS
        
        if (estimatingType == "ProbabilityPredict" and len(throughputHistoryLog) > 0 ):
            # throughputEstimate =  ECMModel.probest( C_iMinus1=throughputHistoryLog[-1], binsMe=binsMe, probModel=probabilityModel, marginal = marginalSample )[0]
            throughputEstimate = ( 1 + pGamma/r_i ) * utils.veryConfidentFunction(binsMe=binsMe,probability=probabilityModel,past= indexPast , quant=pEpsilon)[0]
            suggestedFrameSize = throughputEstimate * T_i

            if (throughputEstimate == -1 and singleFrame!=0):
                suggestedFrameSize = T_i * mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ])
                        
        elif (estimatingType == "A.M." and len(throughputHistoryLog) > 0 ):
            suggestedFrameSize = T_i * mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ])

        # need to judge in my new rule
        # if suggested value < minimal size, then we discard the frame
        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize )

        # Until now, the suggestedFrameSize is fixed.
        ########################################################################################

        # We record the sent frames' information in this array.
        realVideoFrameSize.append(thisFrameSize)
        # readVideoFrameNo.append(singleFrame)

        # frame_start_send_time is the list of UPLOADER's sending time on each frame
        frame_start_send_time.append(runningTime)

        # The following function will calculate t_f.
        uploadFinishTime = utils.packet_level_frame_upload_finish_time( runningTime= runningTime,
                                                                        packet_level_data= networkEnvPacket,
                                                                        packet_level_timestamp= networkEnvTime,
                                                                        framesize= thisFrameSize)

        uploadDuration = uploadFinishTime - runningTime
        # To update the current time clock, now we finished the old frame's transmission.
        runningTime = runningTime + uploadDuration 

        # Here we calculated the C_{i-1}
        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)

        if (len(throughputHistoryLog)>0 and estimatingType == "ProbabilityPredict"):
                self = -1
                past = -1

                for indexSelf in range(len(binsMe)-1): 
                    if (binsMe[indexSelf] <= throughputEstimate and binsMe[indexSelf+1] >throughputEstimate):
                        self = indexSelf
                    
                for indexPast in range(len(binsMe)-1):
                    if (binsMe[indexPast] <= throughputHistoryLog[-1] and binsMe[indexPast+1] > throughputHistoryLog[-1] ):
                        past = indexPast

                probabilityModel[past][self] += 1
                
                toBeDeleted.append(past)
                toBeDeleted.append(self)

                if (forTrain == False):
                    probabilityModel[toBeDeleted[0]][toBeDeleted[1]] -= 1
                    toBeDeleted = toBeDeleted[2:]

    return [ sum(realVideoFrameSize), probabilityModel, count_skip, minimal_framesize, len(realVideoFrameSize)]



number = 25

mAxis = [1,16,128]
xAxis =  np.linspace(0.01, 0.25 ,num=number, endpoint=True)

# To Train the Model
pre = utils.constructProbabilityModel( networkEnvBW = sampleThroughputRecord,  
                                       binsMe = binsMe,  
                                       networkSampleFreq = 1/FPS,  
                                       traceDataSampleFreq = 1/FPS )

model_trained = pre[0]
forgetList = pre[1]

# for i in range( floor(samplePoints/2) ,floor(samplePoints/2)+5):
#         origData = utils.mleFunction(binsMe=binsMe , probability=model_trained, past= i)
#         y =  origData[-1]
#         ag, bg = laplace.fit( y )

#         pyplot.hist(y,bins=binsMe,density=False)
#         binUsed = [0] + binsMe
#         # pyplot.plot(binsMe, 
#         #             [ len(y)*( laplace.cdf(binUsed[min(v+1,len(binUsed)-1)], ag, bg) -laplace.cdf(binUsed[v], ag, bg) ) for v in range(len(binUsed))], 
#         #             '--', 
#         #             color ='black')
#         pyplot.xlabel("Sampled Ci's magnitude")
#         pyplot.ylabel("# of occurrence")
#         pyplot.show()

df = pd.DataFrame(model_trained).to_csv("da.csv",header=False,index=False)

toPlot = 0

for trackUsed in mAxis:
    y1Axis = []
    y2Axis = []
    y3Axis = []
    z1Axis = []
    z2Axis = []

    for x in xAxis:
        a = uploadProcess('dummyUsername1', x , "A.M.", "dummy", forTrain=False, pTrackUsed=trackUsed, pForgetList=[])
        b = uploadProcess('dummyUsername2', x , "ProbabilityPredict", model_trained , forTrain=False, pTrackUsed=trackUsed, pForgetList=forgetList)
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
    pyplot.legend( ["Empirical Condt'l", "A.M. M=" + str(trackUsed),], loc=2)

    toPlot += 1
    pyplot.subplot( len(mAxis),2,toPlot)
    pyplot.xlabel("Minimal Each Frame Size (in MB)")
    pyplot.ylabel("Data sent in" + str(howLongIsVideoInSeconds) +"sec" )
    pyplot.plot(xAxis, z2Axis, '-s', color='blue',
                markersize=1, linewidth=1)
    pyplot.plot(xAxis, z1Axis, '-s', color='red',
                markersize=1, linewidth=1)
    pyplot.legend( ["Empirical Condt'l", "A.M. M=" + str(trackUsed),], loc=4)

pyplot.show()