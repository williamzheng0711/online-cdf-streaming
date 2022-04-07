# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Mar. 9, 2022

import numpy as np
from numpy.core.fromnumeric import argmax, mean, size, var
import utils as utils
import matplotlib.pyplot as pyplot
from numpy import linalg as LA, quantile


howmany_Bs_IN_1Mb = 1024*1024/8

whichVideo = 13
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
                networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
                networkEnvTime_AM.append(nowFileTime - initialTime)
                networkEnvPacket_AM.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
                if (len(packet_level_integral_C_training)==0):
                    packet_level_integral_C_training.append(0 )
                else:
                    packet_level_integral_C_training.append(networkEnvPacket[-1]+packet_level_integral_C_training[-1])
                packet_level_time_training.append(nowFileTime - initialTime)
                count = count  +1 
            elif (nowFileTime - initialTime) < 5*(PreRunTime):
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
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
print(mean(sampleThroughputRecord))
print("This is mean above (Mbps)")

pEpsilon = 0.05
testingTimeStart = timeTrack

def uploadProcess( minimal_framesize, estimatingType, pLogCi, pTrackUsed, 
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
        if (singleFrame % (25 * FPS) == 0 and estimatingType == "ProbabilityPredict"):
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
        T_i = max( (1/FPS - delta),0 )
        
        throughputHistoryLog = throughputHistoryLog[ max((len(throughputHistoryLog) -1 - lenLimit),0) : len(throughputHistoryLog)]
        if (estimatingType == "ProbabilityPredict"):
            localLenLimit = 60 * FPS
            lookBackwardHistogramC = utils.generatingBackwardHistogram(FPS=FPS, int_C=packet_level_integral_C,
                                                                        timeSeq=packet_level_time,
                                                                        currentTime=runningTime, 
                                                                        lenLimit = localLenLimit) 
            decision_list = lookBackwardHistogramC[ max((len(lookBackwardHistogramC) -1 - lenLimit),0) : len(lookBackwardHistogramC)]
            C_iMinus1 = decision_list[-1]

            subLongSeq = [
                decision_list[i+1] 
                for _, i in 
                    zip(decision_list,range(len(decision_list))) 
                if ( (abs((decision_list[i]-C_iMinus1))/C_iMinus1<= 0.05 ) and  i<len(decision_list)-1 ) ]
            # print("C_i-1 =" +str(C_iMinus1) + " ///// " + str(mean(decision_list)) +" | "+ str(mean(subLongSeq)) + " | " + str(mean(throughputHistoryLog)) ) 
                    
            try: 
                if (len(subLongSeq)>30):
                    quantValue = quantile(subLongSeq, pEpsilon)
                    throughputEstimate = quantValue * (FPS*(max(timeBuffer + T_i, 1/FPS) ) )
                    suggestedFrameSize = throughputEstimate  * (1/FPS)
                    # if (runningTime - testingTimeStart> 0):
                    #     print(C_iMinus1)
                    #     pyplot.hist(subLongSeq, bins=200)
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
        # print(thisFrameSize/uploadDuration)

        timeBuffer = max ( pBufferTime - max(runningTime - frame_prepared_time[singleFrame  ],0 ) , 0 ) 

        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)

    return [ sum(realVideoFrameSize), [], count_skip, minimal_framesize, len(realVideoFrameSize)]


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


colorList = ["red", "orange", "purple"]
bufferSizeArray = np.arange(1, 10, step = 1)
Cond_Lossrate = []
Cond_Bitrate = []
Minimal_Lossrate = []
Minimal_Bitrate = []

a_small_minimal_framesize = 0.0001

mAxis = [5,16,128]
for bufferTime in bufferSizeArray:
    ConditionalProposed = uploadProcess(
                        minimal_framesize= a_small_minimal_framesize, 
                        estimatingType = "ProbabilityPredict", 
                        pLogCi = bigHistorySequence , 
                        pTrackUsed=0, 
                        packet_level_integral_C=packet_level_integral_C_original, 
                        packet_level_time=packet_level_time_original,
                        pBufferTime = bufferTime/FPS)

    count_skip_conditional = ConditionalProposed[2]
    Cond_Lossrate.append(count_skip_conditional/(howLongIsVideoInSeconds*FPS))
    Cond_Bitrate.append(ConditionalProposed[0]/howLongIsVideoInSeconds)
    print("Cond'l Proposed Method. Bitrate: " + str(ConditionalProposed[0]/howLongIsVideoInSeconds) + 
        " (Mbps). Loss rate: " + str(count_skip_conditional/(howLongIsVideoInSeconds*FPS)) )

    MinimalFrameScheme = uploadProcess(
                        minimal_framesize = a_small_minimal_framesize, 
                        estimatingType = "MinimalFrame", 
                        pLogCi = bigHistorySequence, 
                        pTrackUsed=0, 
                        packet_level_time=[], 
                        packet_level_integral_C=[],
                        pBufferTime = bufferTime/FPS)

    count_skip_minimal = MinimalFrameScheme[2]
    Minimal_Lossrate.append(count_skip_minimal/(howLongIsVideoInSeconds*FPS))
    Minimal_Bitrate.append(MinimalFrameScheme[0] / howLongIsVideoInSeconds )
    print("Minimal Framesize Method. Bitrate: " + str(MinimalFrameScheme[0] / howLongIsVideoInSeconds) + 
        " (MB/s). Loss rate: " + str(count_skip_minimal/(howLongIsVideoInSeconds*FPS)))

    print("-------------------------------------------------")


AM_LossRateMatrix = [ [0] * len(bufferSizeArray)  for _ in range(len(mAxis))]
AM_BitrateMatrix  = [ [0] * len(bufferSizeArray)  for _ in range(len(mAxis))]
for trackUsed, ix in zip(mAxis,range(len(mAxis))):
    for bufferTimeInArray, iy in zip(bufferSizeArray, range(len(bufferSizeArray))):
        
        Arithmetic_Mean = uploadProcess( 
                            minimal_framesize = a_small_minimal_framesize, 
                            estimatingType = "A.M.", 
                            pLogCi = bigHistorySequence, 
                            pTrackUsed=trackUsed, 
                            packet_level_time=[],
                            packet_level_integral_C=[],
                            pBufferTime = bufferTimeInArray/FPS)

        count_skip_AM = Arithmetic_Mean[2]
        AM_LossRateMatrix[ix][iy] = count_skip_AM/(howLongIsVideoInSeconds*FPS)
        AM_BitrateMatrix[ix][iy] = Arithmetic_Mean[0] / howLongIsVideoInSeconds 
        print("Arithmetic Mean. Bitrate :(M = "+str(trackUsed)+"): " + str(Arithmetic_Mean[0]/howLongIsVideoInSeconds) + 
            " (MB/s). Loss rate: " + str(count_skip_AM/(howLongIsVideoInSeconds*FPS)))


pyplot.xlabel("Bitrate (in MB/s)")
pyplot.ylabel("Loss rate")
# for ix in range(len(mAxis)):
#     pyplot.plot(AM_BitrateMatrix[ix], AM_LossRateMatrix[ix], '-s', color=colorList[ix], markersize=1, linewidth=1)
# pyplot.plot(Minimal_Bitrate, Minimal_Lossrate, '-s', color = "black", markersize = 1, linewidth = 1)
pyplot.plot(Cond_Bitrate, Cond_Lossrate, '-s', color='blue',markersize=1, linewidth=1)

for x_axis,y_axis,bufferInteger in zip(Cond_Bitrate, Cond_Lossrate,bufferSizeArray):
    pyplot.annotate(str(bufferInteger), (float(x_axis), float(y_axis)) )

AMLegend = ["A.M. M=" + str(trackUsed) for trackUsed in mAxis]
pyplot.legend(AMLegend + 
            ["Fixed as Minimal"] +
            ["Empirical Condt'l"], 
            loc="best")

pyplot.show()