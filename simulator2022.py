# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Nov. 7, 2021

import csv
from math import exp, floor, pi, sin, sqrt
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

import ecm_model as ECMModel

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000.0*1000.0
FPS = 25

networkSamplingInterval = 0.04

count = 0
howLongIsVideo = 10000

NETWORK_TRACE = "1h_less"
network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'

networkEnvTime = []
networkEnvTP= []

timeDataLoad = 20000

whichVideo = 12
for suffixNum in range(whichVideo,whichVideo+1):
    networkEnvTP = []
    with open( network_trace_dir+str(suffixNum) + ".csv" ) as file1:
        for line in file1:
            count = count  +1 
            # if (count<400000):
            parse = line.split()
            networkEnvTP.append(float(parse[0]) / B_IN_MB ) 


startPoint = np.quantile(networkEnvTP, 0.005)
endPoint = np.quantile(networkEnvTP, 0.995)
MIN_TP = min(networkEnvTP)
MAX_TP = max(networkEnvTP)

samplePoints = 15
marginalSample = 2
        
if (startPoint!=0):
    binsMe=np.concatenate( (np.linspace( MIN_TP,startPoint, marginalSample, endpoint=False) , 
                            np.linspace( startPoint, endPoint, samplePoints, endpoint=False) ,  
                            np.linspace( endPoint, MAX_TP, marginalSample, endpoint=True)  ), 
                            axis=0)
else:
    binsMe = np.concatenate(( np.linspace( startPoint, endPoint, samplePoints, endpoint=False) ,  
                              np.linspace( endPoint, MAX_TP, marginalSample, endpoint=True)  ), 
                              axis=0)

probability  = [ [0] * len(binsMe)  for _ in range(len(binsMe))]

def uploadProcess(user_id, minimal_framesize, estimatingType, probability, forTrain, pTrackUsed, pForgetList):
    frame_prepared_time = []
    frame_start_send_time = []
    throughputHistoryLog = []
    realVideoFrameSize = []
    readVideoFrameNo = []
    probabilityModel = np.array(probability)
    toBeDeleted = pForgetList
    if (forTrain):
        timeNeeded = timeDataLoad
    else: 
        timeNeeded = howLongIsVideo

    tempTimeTrack = (1/FPS)*timeDataLoad
    for _ in range(timeNeeded):
        frame_prepared_time.append(tempTimeTrack)
        tempTimeTrack = tempTimeTrack + 1/FPS
    

    uploadDuration = 0
    # This ensures that training data is not over-lapping with testing data
    runningTime = (1/FPS)*timeDataLoad 

    # initialize the C_0 hat (in MB)
    throughputEstimate = (1/FPS) * mean(networkEnvTP) 
    count_skip = 0

    # Note that the frame_prepared_time every time is NON-SKIPPABLE
    for singleFrame in range( timeNeeded ):
        if (singleFrame % 5000 == 0 and forTrain == True): print(str(format(singleFrame/timeNeeded, ".3f"))+ " Please wait..." )
        # The "if" condition is a must
        # because I do not know the corresponding network environment (goes beyond the record of bandwidth).
        if ( int(runningTime / networkSamplingInterval)  >= len(networkEnvTP)
            or singleFrame>timeNeeded ):
            break 

        # To consider the latency caused by previous frames
        if ( singleFrame < timeNeeded-1 and runningTime > frame_prepared_time[singleFrame + 1] + 0.05/FPS):
            count_skip = count_skip + 1
            continue

        if (singleFrame >0 and ( runningTime <= frame_prepared_time[singleFrame])): 
            # 上一次的傳輸太快了，導致新的幀還沒生成出來
            runningTime = frame_prepared_time[singleFrame] 


        delta = runningTime -  frame_prepared_time[singleFrame]
        suggestedFrameSize = minimal_framesize
        if (estimatingType == "ProbabilityPredict" and len(throughputHistoryLog) > 0 ):
            # throughputEstimate =  ECMModel.probest( C_iMinus1=throughputHistoryLog[-1], binsMe=binsMe, probModel=probabilityModel, marginal = marginalSample )[0]
            throughputEstimate = ECMModel.confidenceSuggest(C_iMinus1=throughputHistoryLog[-1], binsMe=binsMe, z=(1/FPS - delta), FPS=FPS, probModel=probabilityModel)[0]
            suggestedFrameSize = throughputEstimate * (1/FPS - delta) 

            if (throughputEstimate == -1 and singleFrame!=0):
                suggestedFrameSize = mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ]) * (1/FPS  - delta) 
                        
        elif (estimatingType == "LastMeasure" and len(throughputHistoryLog) > 0 ):
            suggestedFrameSize = mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ]) * (1/FPS - delta) 

        # need to judge in my new rule
        # if suggested value < minimal size, then we discard the frame
        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize )

        realVideoFrameSize.append(thisFrameSize)
        readVideoFrameNo.append(singleFrame)

        uploadFinishTime = utils.frame_upload_done_time( 
            runningTime = runningTime,
            networkEnvBW = networkEnvTP,
            size = thisFrameSize,
            networkSamplingInterval = networkSamplingInterval)
        
        uploadDuration = uploadFinishTime - runningTime # upload 指的僅僅是把視頻塊塊搬上鏈路
        
        # frame_start_send_time is the list of UPLOADER's sending time on each frame
        frame_start_send_time.append(runningTime)
        runningTime = runningTime + uploadDuration 

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

    return [
        sum(realVideoFrameSize),
        probabilityModel,
        count_skip, 
        minimal_framesize
        ]



number = 50

mAxis = [1,16,128]
xAxis =  np.linspace(0.01, 0.35 ,num=number, endpoint=True)

pre = utils.constructProbabilityModel(
    networkEnvBW=networkEnvTP[0:timeDataLoad], 
    binsMe=binsMe, 
    networkSampleFreq=networkSamplingInterval, 
    traceDataSampleFreq=networkSamplingInterval)

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
        a = uploadProcess('dummyUsername1', x , "LastMeasure", "dummy", forTrain=False, pTrackUsed=trackUsed, pForgetList=[])
        b = uploadProcess('dummyUsername2', x , "ProbabilityPredict", model_trained , forTrain=False, pTrackUsed=trackUsed, pForgetList=forgetList)
        count_skipA = a[2]
        count_skipB = b[2]
        y1Axis.append(count_skipA/howLongIsVideo)
        y2Axis.append(count_skipB/howLongIsVideo)

        z1Axis.append(a[0])
        z2Axis.append(b[0])

        print("LastMeasure: " + str(a[0]) + " " + str(count_skipA/howLongIsVideo) + " with min-size: " + str(a[3]) )
        print("ProbEstTest: " + str(b[0]) + " " + str(count_skipB/howLongIsVideo))


    print("Mean of this network: " + str(mean(networkEnvTP)))
    print("Var of ~: " + str(var(networkEnvTP)))


    toPlot += 1
    pyplot.subplot( len(mAxis),2,toPlot)
    pyplot.xlabel("Minimal Each Frame Size (in MB)")
    pyplot.ylabel("Loss Rate (Discard Rate)")
    pyplot.plot(xAxis, y2Axis, '-s', color='blue', markersize=1, linewidth=1)
    pyplot.plot(xAxis, y1Axis, '-s', color='red', markersize=1, linewidth=1)
    pyplot.legend( ["Condt'l Mean", "A.M. M=" + str(trackUsed),], loc=2)

    toPlot += 1
    pyplot.subplot( len(mAxis),2,toPlot)
    pyplot.xlabel("Minimal Each Frame Size (in MB)")
    pyplot.ylabel("Data sent in" + str(howLongIsVideo/FPS) +"sec" )
    pyplot.plot(xAxis, z2Axis, '-s', color='blue',
                markersize=1, linewidth=1)
    pyplot.plot(xAxis, z1Axis, '-s', color='red',
                markersize=1, linewidth=1)
    pyplot.legend( ["Condt'l Mean", "A.M. M=" + str(trackUsed),], loc=4)

pyplot.show()




