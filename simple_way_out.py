# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Mar. 9, 2022

from cmath import e
import csv
from math import exp, floor, pi, sin, sqrt
from multiprocessing.context import ForkProcess
from operator import index, indexOf, ne
from struct import pack
from matplotlib.axis import Axis
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
import multiLinreg as MLR



B_IN_MB = 1024*1024

whichVideo = 7
FPS = 60

# Testing Set Size
howLongIsVideoInSeconds = 120

network_trace_dir = './dataset/fyp_lab/'

networkEnvTime = []
networkEnvPacket= []
count = 0
initialTime = 0
packet_level_integral_C_training = []
packet_level_time_training = []

networkEnvTime_AM = []
networkEnvPacket_AM= []

PreRunTime = 300

# load the mock data from our local dataset
for suffixNum in range(whichVideo,whichVideo+1):
    with open( network_trace_dir+ str(suffixNum) + ".txt" ) as traceDateFile:
        for eachLine in traceDateFile:
            parse = eachLine.split()
            if (count==0):
                initialTime = float(parse[0])
            nowFileTime = float(parse[0]) 
            if (nowFileTime - initialTime) < PreRunTime:
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / B_IN_MB ) 
                networkEnvTime_AM.append(nowFileTime - initialTime)
                networkEnvPacket_AM.append( float(parse[1]) / B_IN_MB ) 
                if (len(packet_level_integral_C_training)==0):
                    packet_level_integral_C_training.append(0 )
                else:
                    packet_level_integral_C_training.append(networkEnvPacket[-1]+packet_level_integral_C_training[-1])
                packet_level_time_training.append(nowFileTime - initialTime)
                count = count  +1 
            elif (nowFileTime - initialTime) < 5*(PreRunTime):
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / B_IN_MB ) 
                count = count  +1 
                
print("Before using time-packet, is time order correct? "+ str(packet_level_time_training == sorted(packet_level_time_training, reverse=False)))
print("Before using integral records, is the order correct? "+ str(packet_level_integral_C_training == sorted(packet_level_integral_C_training, reverse=False)))


############################################################################
# To train the data set used for AM algorithm
timeTrack = 0
amount = 0
sampleThroughputRecord = []
for numberA in range(len(networkEnvPacket_AM)):
    amount = amount + networkEnvPacket[numberA]
    if ( ( networkEnvTime[numberA] - timeTrack ) > 1 / FPS ):
        throughputLast = amount / ( networkEnvTime[numberA] - timeTrack  )
        timeTrack = networkEnvTime[numberA]
        sampleThroughputRecord.append( throughputLast )
        amount = 0

############################################################################

pEpsilon = 0.05
testingTimeStart = timeTrack

def uploadProcess(user_id, minimal_framesize, estimatingType, pLogCi, forTrain, pTrackUsed, pForgetList, 
                    packet_level_integral_C, packet_level_time, pBufferTime):

    timeBuffer = pBufferTime
    packet_level_integral_C_inside = packet_level_integral_C
    packetLevelTimeInside = packet_level_time


    frame_prepared_time = []
    throughputHistoryLog = pLogCi[ max(0, len(pLogCi) -1 - lenLimit) : len(pLogCi)]

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
        if (singleFrame % (20 * FPS) == 0 and estimatingType == "ProbabilityPredict"):
            print("Now is time: " + str(runningTime - testingTimeStart))
        ########################################################################################

        if (runningTime - testingTimeStart > howLongIsVideoInSeconds):
            break 

        
        if (singleFrame >0 and ( runningTime < frame_prepared_time[singleFrame])): 
            # 上一次的傳輸太快了，導致新的幀還沒生成出來
            # Then we need to wait until singleframe is generated and available to send.
            runningTime = frame_prepared_time[singleFrame]
        
        if (runningTime - frame_prepared_time[singleFrame] > 1/FPS + timeBuffer):
            count_skip = count_skip + 1
            timeBuffer = max ( pBufferTime - max(runningTime - frame_prepared_time[singleFrame  ],0 ) , 0 ) 
            continue

        #######################################################################################
        # Anyway, from now on, the uploader is ready to send singleFrame
        # In this part, we determine what is the suggestedFrameSize 
        # We initialize the guess as minimal_framesize
        suggestedFrameSize = -np.Infinity

        delta = runningTime -  frame_prepared_time[singleFrame]
        T_i = (1/FPS - delta)
        
        throughputHistoryLog = throughputHistoryLog[ max((len(throughputHistoryLog) -1 - lenLimit),0) : len(throughputHistoryLog)]
        if (estimatingType == "ProbabilityPredict"):
            localLenLimit = 60 * FPS
            lookBackwardHistogramC = utils.generatingBackwardHistogram(FPS=FPS, int_C=packet_level_integral_C,
                                                                        timeSeq=packet_level_time,
                                                                        currentTime=runningTime, 
                                                                        lenLimit = localLenLimit) 
            assemble_list = lookBackwardHistogramC
            decision_list = assemble_list[ max((len(assemble_list) -1 - lenLimit),0) : len(assemble_list)]
            C_iMinus1 = decision_list[-1]

            subLongSeq = [
                decision_list[i+1] 
                for _, i in 
                    zip(decision_list,range(len(decision_list))) 
                if ( (abs((decision_list[i]-C_iMinus1))/C_iMinus1<= 0.05 ) and  i<len(decision_list)-1 ) ]
                    
            try: 
                if (len(subLongSeq)>30):
                    quantValue = quantile(subLongSeq, pEpsilon)
                    throughputEstimate = quantValue * (FPS*(max(timeBuffer + T_i, 1/FPS) ) )
                    suggestedFrameSize = throughputEstimate  * (1/FPS)
                    # if (runningTime - testingTimeStart> 0):
                    #     print(C_iMinus1)
                    #     pyplot.hist(subLongSeq, bins=50)
                    #     pyplot.show()
                else:
                    quantValue = quantile(decision_list, pEpsilon)
                    throughputEstimate = quantValue * (FPS*(max(timeBuffer + T_i, 1/FPS) ) )
                    suggestedFrameSize = throughputEstimate  * (1/FPS)
            except:
                suggestedFrameSize = minimal_framesize

        elif (estimatingType == "A.M." and len(throughputHistoryLog) > 0 ):
            try:
                adjustedAM_Nume = sum(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)])
                adjustedAM_Deno = [ a/b for a,b in zip(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)], 
                                        throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog)]) ]
                C_i_hat_AM = adjustedAM_Nume/sum(adjustedAM_Deno)
                suggestedFrameSize = (1/FPS) * C_i_hat_AM
            except:
                suggestedFrameSize = (1/FPS) * mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ])
        
        elif (estimatingType == "MinimalFrame"):
            suggestedFrameSize = minimal_framesize
        
        elif (estimatingType == "Marginal"):
            lookBackwardHistogramC = utils.generatingBackwardHistogram(FPS=FPS, int_C=packet_level_integral_C,timeSeq=packet_level_time,currentTime=runningTime, lenLimit = lenLimit) 
            assemble_list = throughputHistoryLog + lookBackwardHistogramC
            decision_list = assemble_list[ max((len(throughputHistoryLog) -1 - lenLimit),0) : len(throughputHistoryLog)]
            suggestedFrameSize = quantile(decision_list, pEpsilon) * (1/FPS)
        # elif (estimatingType == "OLS"):
        #     pastDataUse = 30
        #     explantoryRVs = throughputHistoryLog[ len(throughputHistoryLog)-pastDataUse : len(throughputHistoryLog)][::-1]
        #     explantoryRVs.insert(0,1)
        #     coef = MLR.MLS(timeSeries= throughputHistoryLog, predictorsNum= pastDataUse, miniTrainingSize= 10)
        #     C_i_guess_MLR = np.array(explantoryRVs).dot(coef)  
        #     suggestedFrameSize = C_i_guess_MLR * T_i
            # print("suggestedFrameSize of OLS: " + str(suggestedFrameSize))

        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize )

        # Until now, the suggestedFrameSize is fixed.
        #######################################################################################
        [uploadFinishTime,packet_level_integral_C_inside, packetLevelTimeInside ] = utils.packet_level_frame_upload_finish_time( 
                                                                        runningTime= runningTime,
                                                                        packet_level_data= networkEnvPacket,
                                                                        packet_level_timestamp= networkEnvTime,
                                                                        framesize= thisFrameSize,
                                                                        toUsePacketRecords = (estimatingType == "ProbabilityPredict") or (estimatingType =="Marginal"),
                                                                        packet_level_integral_C = packet_level_integral_C_inside,
                                                                        packet_level_time = packetLevelTimeInside,)

        # We record the sent frames' information in this array.
        if (uploadFinishTime<=howLongIsVideoInSeconds + testingTimeStart):
            realVideoFrameSize.append(thisFrameSize)

        uploadDuration = uploadFinishTime - runningTime
        runningTime = runningTime + uploadDuration 

        timeBuffer = max ( pBufferTime - max(runningTime - frame_prepared_time[singleFrame  ],0 ) , 0 ) 

        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)

    return [ sum(realVideoFrameSize), [], count_skip, minimal_framesize, len(realVideoFrameSize)]



number = 3

mAxis = [5,16,128]
xAxis =  np.linspace(0.000005, 0.015 ,num=number, endpoint=True)

lenLimit = PreRunTime * FPS
bigHistorySequence = sampleThroughputRecord[ max((len(sampleThroughputRecord)-lenLimit),0):len(sampleThroughputRecord)]

ecmLossRateArray = []
ecmTotalSizeArray = []

minimalLossRateArray = []
minimalSizeArray = []

marginalProbLossRateArray = []
marginalProbSizeArray = []

OLSpredLossRateArray = []
OLSpredSizeArray = []

packet_level_integral_C_original = packet_level_integral_C_training
packet_level_time_original = packet_level_time_training

for x_for_b in xAxis:

    b = uploadProcess('dummyUsername2', x_for_b , "ProbabilityPredict", 
                        pLogCi=bigHistorySequence , forTrain=False, pForgetList=[], 
                        pTrackUsed=0, 
                        packet_level_integral_C=packet_level_integral_C_original, 
                        packet_level_time=packet_level_time_original,
                        pBufferTime = 1/FPS)
    count_skipB = b[2]
    ecmLossRateArray.append(count_skipB/(howLongIsVideoInSeconds*FPS))
    ecmTotalSizeArray.append(b[0])
    print("OurMethod: " + str(b[0]) + " " + str(count_skipB/(howLongIsVideoInSeconds*FPS)))

    m = uploadProcess('dummyUsername2', x_for_b , "MinimalFrame", 
                        pLogCi=bigHistorySequence , forTrain=False, pForgetList=[], 
                        pTrackUsed=0, 
                        packet_level_time=[], packet_level_integral_C=[],
                        pBufferTime = 1/FPS)
    count_skipM = m[2]
    minimalLossRateArray.append(count_skipM/(howLongIsVideoInSeconds*FPS))
    minimalSizeArray.append(m[0])
    print("Minimal: " + str(m[0]) + " " + str(count_skipM/(howLongIsVideoInSeconds*FPS)))

    print("----------------------------")

amLossRateMatrix = [ [0] * len(xAxis)  for _ in range(len(mAxis))]
amTotalSizeMatrix  = [ [0] * len(xAxis)  for _ in range(len(mAxis))]
for trackUsed, ix in zip(mAxis,range(len(mAxis))):
    for x, iy in zip(xAxis, range(len(xAxis))):
        a = uploadProcess('dummyUsername1', x , "A.M.", pLogCi=bigHistorySequence, forTrain=False, pTrackUsed=trackUsed, pForgetList=[], 
                            packet_level_time=[],packet_level_integral_C=[],
                            pBufferTime= 0)
        count_skipA = a[2]
        amLossRateMatrix[ix][iy] = count_skipA/(howLongIsVideoInSeconds*FPS)
        amTotalSizeMatrix[ix][iy] = a[0]
        print("A.M.(M="+str(trackUsed)+"): " +  str(a[0]) + " " + str(count_skipA/(howLongIsVideoInSeconds*FPS)) + " with min-size: " + str(a[3]) )


colorList = ["red", "orange", "purple"]

pyplot.subplot( 2,2,1)
pyplot.xlabel("Minimal Each Frame Size (in MB)")
pyplot.ylabel("Loss Rate")
for ix in range(len(mAxis)):
    pyplot.plot(xAxis, amLossRateMatrix[ix], '-s', color=colorList[ix], markersize=1, linewidth=1)
pyplot.plot(xAxis, minimalLossRateArray, '-s', color='black', markersize=1, linewidth=1)
AMLegendLossRate = ["A.M. M=" + str(trackUsed) for trackUsed in mAxis]
pyplot.plot(xAxis, ecmLossRateArray, '-s', color='blue', markersize=1, linewidth=1)
pyplot.legend(   AMLegendLossRate + 
                ["Fixed as Minimal"] +
                ["Empirical Condt'l"]
                , loc="best")


pyplot.subplot( 2,2,2)
pyplot.xlabel("Minimal Each Frame Size (in MB)")
pyplot.ylabel("Data sent in" + str(howLongIsVideoInSeconds) +" sec" )
for ix in range(len(mAxis)):
    pyplot.plot(xAxis, amTotalSizeMatrix[ix], '-s', color=colorList[ix], markersize=1, linewidth=1)
pyplot.plot(xAxis, minimalSizeArray, '-s', color='black',markersize=1, linewidth=1)
AMLegendTotalSize = ["A.M. M=" + str(trackUsed) for trackUsed in mAxis]
pyplot.plot(xAxis, ecmTotalSizeArray, '-s', color='blue',markersize=1, linewidth=1)
pyplot.legend(  AMLegendTotalSize + 
                ["Fixed as Minimal"] +
                ["Empirical Condt'l"] 
                ,loc="best")

pyplot.title("Target: " + str(pEpsilon))




# bufferSizeArray = np.arange(0, FPS, step = 1)
bufferSizeArray = [0,1,2,3,4,5,6]
ECM_LR_vs_BT = []
ECM_Size_vs_BT = []
Min_LR_vs_BT = []
Min_Size_vs_BT = []

for bufferTime in bufferSizeArray:
    ourMethod = uploadProcess('dummyUsername2', xAxis[0] , "ProbabilityPredict", 
                        pLogCi=bigHistorySequence , forTrain=False, pForgetList=[], 
                        pTrackUsed=0, 
                        packet_level_integral_C=packet_level_integral_C_original, 
                        packet_level_time=packet_level_time_original,
                        pBufferTime = bufferTime/FPS)
    count_skipB = ourMethod[2]
    ECM_LR_vs_BT.append(count_skipB/(howLongIsVideoInSeconds*FPS))
    ECM_Size_vs_BT.append(ourMethod[0])
    print("OurMethod vs buffer: " + str(ourMethod[0]) + " " + str(count_skipB/(howLongIsVideoInSeconds*FPS)))

    MinimalFrameMethod = uploadProcess('dummyUsername2', xAxis[0] , "MinimalFrame", 
                        pLogCi=bigHistorySequence , forTrain=False, pForgetList=[], 
                        pTrackUsed=0, 
                        packet_level_time=[], 
                        packet_level_integral_C=[],
                        pBufferTime = bufferTime/FPS)
    count_skipM = MinimalFrameMethod[2]
    Min_LR_vs_BT.append(count_skipM/(howLongIsVideoInSeconds*FPS))
    Min_Size_vs_BT.append(MinimalFrameMethod[0])
    print("Minimal: " + str(MinimalFrameMethod[0]) + " " + str(count_skipM/(howLongIsVideoInSeconds*FPS)))
    print("----------------------------")

amLossRateMatrix_vs_buffer = [ [0] * len(bufferSizeArray)  for _ in range(len(mAxis))]
amTotalSizeMatrix_vs_buffer  = [ [0] * len(bufferSizeArray)  for _ in range(len(mAxis))]
for trackUsed, ix in zip(mAxis,range(len(mAxis))):
    for bufferTimeArray, iy in zip(bufferSizeArray, range(len(bufferSizeArray))):
        AM_vs_buffer = uploadProcess('dummyUsername1', 
                            xAxis[0] , "A.M.", 
                            pLogCi=bigHistorySequence, forTrain=False, 
                            pTrackUsed=trackUsed, pForgetList=[], 
                            packet_level_time=[],
                            packet_level_integral_C=[],
                            pBufferTime = bufferTimeArray/FPS)
        count_skipA = AM_vs_buffer[2]
        amLossRateMatrix_vs_buffer[ix][iy] = count_skipA/(howLongIsVideoInSeconds*FPS)
        amTotalSizeMatrix_vs_buffer[ix][iy] = AM_vs_buffer[0]
        print("A.M.(M="+str(trackUsed)+"): " +  str(AM_vs_buffer[0]) + " " + str(count_skipA/(howLongIsVideoInSeconds*FPS)) + " with min-size: " + str(AM_vs_buffer[3]) )


pyplot.subplot( 2,2,3)
pyplot.xlabel("Buffer Time: in 1/FPS")
pyplot.ylabel("Loss Rate")
for ix in range(len(mAxis)):
    pyplot.plot(bufferSizeArray, amLossRateMatrix_vs_buffer[ix], '-s', color=colorList[ix], markersize=1, linewidth=1)
pyplot.plot(bufferSizeArray, Min_LR_vs_BT, '-s', color='black', markersize=1, linewidth=1)
AMLegendLossRate = ["A.M. M=" + str(trackUsed) for trackUsed in mAxis]
pyplot.plot(bufferSizeArray, ECM_LR_vs_BT, '-s', color='blue', markersize=1, linewidth=1)
pyplot.legend(   AMLegendLossRate + 
                ["Fixed as Minimal"] +
                ["Empirical Condt'l"]
                , loc="best")


pyplot.subplot( 2,2,4)
pyplot.xlabel("Buffer Time (in second)")
pyplot.ylabel("Data sent in" + str(howLongIsVideoInSeconds) +" sec" )
for ix in range(len(mAxis)):
    pyplot.plot(bufferSizeArray, amTotalSizeMatrix_vs_buffer[ix], '-s', color=colorList[ix], markersize=1, linewidth=1)
pyplot.plot(bufferSizeArray, Min_Size_vs_BT, '-s', color='black',markersize=1, linewidth=1)
AMLegendTotalSize = ["A.M. M=" + str(trackUsed) for trackUsed in mAxis]
pyplot.plot(bufferSizeArray, ECM_Size_vs_BT, '-s', color='blue',markersize=1, linewidth=1)
pyplot.legend(  AMLegendTotalSize + 
                ["Fixed as Minimal"] +
                ["Empirical Condt'l"] 
                ,loc="best")
pyplot.show()