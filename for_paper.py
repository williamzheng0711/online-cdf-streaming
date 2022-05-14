# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     May 12, 2022


from ctypes import util
from math import floor
from unittest import skip
import numpy as np
from numpy.core.fromnumeric import mean
import utils as utils
import matplotlib.pyplot as pyplot
from numpy import cumsum, quantile

network_trace_dir = './dataset/fyp_lab/'
howmany_Bs_IN_1Mb = 1024*1024/8


FPS = 60
whichVideo = 13
# Testing Set Size
howLongIsVideoInSeconds = 240

networkEnvTime = []
networkEnvPacket= []
count = 0
initialTime = 0
packet_level_integral_C_training = []
packet_level_time_training = []


for suffixNum in range(whichVideo,whichVideo+1):
    with open( network_trace_dir+ str(suffixNum) + ".txt" ) as traceDateFile:
        for eachLine in traceDateFile:
            parse = eachLine.split()
            if (count==0):
                initialTime = float(parse[0])
            nowFileTime = float(parse[0]) 
            networkEnvTime.append(nowFileTime - initialTime)
            networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
            count = count  +1 

throughputEstimateInit = sum(networkEnvPacket[0:10000]) / (networkEnvTime[10000-1])
print( str(throughputEstimateInit) + 
        "Mbps, this is mean throughput")

pEpsilon = 0.05

def uploadProcess( minimal_framesize, estimatingType, pTrackUsed, pBufferTime):
    
    timeBuffer = pBufferTime

    frame_prepared_time = []
    throughputHistoryLog = []
    transmitHistoryTimeLog = []
    transmitHistoryTimeCum = []
    realVideoFrameSize = []

    timeline = np.arange(0, howLongIsVideoInSeconds, step = 0.25)
    skipNumberList = [0] * len(timeline)
    totalNumberList = [0] * len(timeline)

    # This is to SKIP the training part of the data.
    # Hence ensures that training data is not over-lapping with testing data
    # Mock up the frame generating times
    tempTimeTrack = 0
    runningTime = 0
    for _ in range( howLongIsVideoInSeconds * FPS ):
        frame_prepared_time.append(tempTimeTrack)
        tempTimeTrack = tempTimeTrack + 1/FPS
    
    uploadDuration = 0

    throughputEstimate = throughputEstimateInit
    count_skip = 0

    # Note that the frame_prepared_time every time is NON-SKIPPABLE
    for singleFrame in range( howLongIsVideoInSeconds * FPS ):
        totalNumberList[  min(floor(runningTime / 0.25), len(totalNumberList)-1 ) ] += 1
        if (singleFrame % (5 * FPS) == 0 and estimatingType == "ProbabilityPredict"):
            print("Now is time: " + str(runningTime) ) 
        ########################################################################################

        if (runningTime > howLongIsVideoInSeconds):
            break 

        if (singleFrame >0 and  runningTime < frame_prepared_time[singleFrame] ): 
            # Then we need to wait until singleframe is generated and available to send.
            runningTime = frame_prepared_time[singleFrame]
        
        if (runningTime - frame_prepared_time[singleFrame] > 1/FPS + timeBuffer):
            # print("Some frame skipped!")
            count_skip = count_skip + 1
            skipNumberList[ floor(runningTime / 0.25) ] += 1
            timeBuffer = max ( pBufferTime - max(runningTime - frame_prepared_time[singleFrame  ],0 ) , 0 ) 
            continue

        #######################################################################################
        suggestedFrameSize = -np.Infinity

        delta = runningTime -  frame_prepared_time[singleFrame]
        T_i = max( (1/FPS - delta),0 )
        
        if (estimatingType == "ProbabilityPredict"):
            backLen = 1800
            try:
                lookbackwardHistogramS =  utils.generatingBackwardSizeFromLog(
                                            pastDurations= transmitHistoryTimeLog,
                                            pastDurationsCum= transmitHistoryTimeCum,
                                            pastSizes= realVideoFrameSize, 
                                            backLen= backLen,
                                            timeSlot= min(T_i + timeBuffer,1/FPS ),
                                        )
            except:
                lookbackwardHistogramS = []

            # print(len(lookbackwardHistogramS))
            
            if (len(lookbackwardHistogramS)>100):
                decision_list = lookbackwardHistogramS
                Ideal_S_iMinus1 = decision_list[-1]
                subLongSeq = [
                    decision_list[i+1] 
                    for _, i in zip(decision_list,range( len(decision_list) )) 
                        if ( (abs( decision_list[i] / Ideal_S_iMinus1- 1 )<= 0.025 ) and  i< len(decision_list) -1 ) ]
                    
                if (len(subLongSeq)>25):
                    quantValue = quantile(subLongSeq, pEpsilon)
                    suggestedFrameSize = quantValue
                    # pyplot.hist(subLongSeq, bins=60)
                    # pyplot.axvline(x=quantValue, color = "black")
                    # pyplot.show()
                    # pyplot.xlim([np.percentile(decision_list,0), np.percentile(decision_list,99.5)]) 
                    # pyplot.hist(decision_list, bins=1000)
                    # pyplot.show()
                else:
                    quantValue = quantile(decision_list, pEpsilon)
                    suggestedFrameSize = quantValue
            
            else:
                if (len(throughputHistoryLog) > 0 ):
                    try:
                        adjustedAM_Nume = sum(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)])
                        adjustedAM_Deno = [ a/b for a,b in zip(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)], 
                                                throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog)]) ]
                        C_i_hat_AM = adjustedAM_Nume/sum(adjustedAM_Deno)
                        suggestedFrameSize = (1/FPS) * C_i_hat_AM
                    except:
                        suggestedFrameSize = (1/FPS) * mean(throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog) ])


        # elif (estimatingType == "Marginal"):
        #     backRange = 20          
        #     lookbackwardHistogramS =  utils.generatingBackwardSizeFromLog(
        #                                 pastDurations= transmitHistoryTimeLog,
        #                                 pastDurationsCum= transmitHistoryTimeCum,
        #                                 pastSizes= realVideoFrameSize, 
        #                                 backwardTimeRange= backRange,
        #                                 timeSlot= T_i + timeBuffer/2,
        #                             )

        #     decision_list = lookbackwardHistogramS
        #     quantValue = quantile(decision_list, pEpsilon)
        #     suggestedFrameSize = quantValue


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
        
        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize )




        # Until now, the suggestedFrameSize is fixed.
        #######################################################################################
        uploadFinishTime = utils.paper_frame_upload_finish_time( 
                                runningTime= runningTime,
                                packet_level_data= networkEnvPacket,
                                packet_level_timestamp= networkEnvTime,
                                framesize= thisFrameSize)


        # We record the sent frames' information in this array.
        if (uploadFinishTime<=howLongIsVideoInSeconds):
            realVideoFrameSize.append(thisFrameSize)

        uploadDuration = uploadFinishTime - runningTime
        runningTime = runningTime + uploadDuration 
        # print(thisFrameSize/uploadDuration)
        # print(runningTime)

        timeBuffer = max ( pBufferTime - max(runningTime - frame_prepared_time[singleFrame  ],0 ) , 0 ) 

        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)
        transmitHistoryTimeLog.append(uploadDuration)
        if (len(transmitHistoryTimeCum)>0):
            transmitHistoryTimeCum.append(transmitHistoryTimeCum[-1]+uploadDuration)
        else:
            transmitHistoryTimeCum.append(uploadDuration)
    

    cumsum_skip = cumsum(skipNumberList)
    cumsum_ttotal = cumsum(totalNumberList)
    pyplot.plot(timeline,[a/b for a,b in zip(cumsum_skip,cumsum_ttotal)])
    pyplot.show()

    return [ sum(realVideoFrameSize), [], count_skip, minimal_framesize, len(realVideoFrameSize)]






colorList = ["red", "orange", "goldenrod"]
bufferSizeArray = np.arange(0, 6.25, step = 2)
Cond_Lossrate = []
Cond_Bitrate = []
Minimal_Lossrate = []
Minimal_Bitrate = []
Marginal_Lossrate = []
Marginal_Bitrate = []

a_small_minimal_framesize = 0.0001

mAxis = [5,16,128]
for bufferTime in bufferSizeArray:
    ConditionalProposed = uploadProcess(
                        minimal_framesize= a_small_minimal_framesize, 
                        estimatingType = "ProbabilityPredict", 
                        pTrackUsed=5, 
                        pBufferTime = bufferTime/FPS)

    count_skip_conditional = ConditionalProposed[2]
    Cond_Lossrate.append(count_skip_conditional/(howLongIsVideoInSeconds*FPS))
    Cond_Bitrate.append(ConditionalProposed[0]/howLongIsVideoInSeconds)
    print("Cond'l Proposed Method. Bitrate: " + str(ConditionalProposed[0]/howLongIsVideoInSeconds) + 
        " (Mbps). Loss rate: " + str(count_skip_conditional/(howLongIsVideoInSeconds*FPS)) )

    MinimalFrameScheme = uploadProcess(
                        minimal_framesize = a_small_minimal_framesize, 
                        estimatingType = "MinimalFrame", 
                        pTrackUsed=0, 
                        pBufferTime = bufferTime/FPS)

    count_skip_minimal = MinimalFrameScheme[2]
    Minimal_Lossrate.append(count_skip_minimal/(howLongIsVideoInSeconds*FPS))
    Minimal_Bitrate.append(MinimalFrameScheme[0] / howLongIsVideoInSeconds )
    print("Minimal Framesize Method. Bitrate: " + str(MinimalFrameScheme[0] / howLongIsVideoInSeconds) + 
        " (Mbps). Loss rate: " + str(count_skip_minimal/(howLongIsVideoInSeconds*FPS)))

    # MarginalScheme = uploadProcess(
    #                     minimal_framesize = a_small_minimal_framesize, 
    #                     estimatingType = "Marginal", 
    #                     pTrackUsed=5, 
    #                     pBufferTime = bufferTime/FPS)

    # count_skip_marginal = MarginalScheme[2]
    # Marginal_Lossrate.append(count_skip_marginal/(howLongIsVideoInSeconds*FPS))
    # Marginal_Bitrate.append(MarginalScheme[0] / howLongIsVideoInSeconds )
    # print("Marginal Framesize Method. Bitrate: " + str(MarginalScheme[0] / howLongIsVideoInSeconds) + 
    #     " (Mbps). Loss rate: " + str(count_skip_marginal/(howLongIsVideoInSeconds*FPS)))

    print("-------------------------------------------------")


AM_LossRateMatrix = [ [0] * len(bufferSizeArray)  for _ in range(len(mAxis))]
AM_BitrateMatrix  = [ [0] * len(bufferSizeArray)  for _ in range(len(mAxis))]
for trackUsed, ix in zip(mAxis,range(len(mAxis))):
    for bufferTimeInArray, iy in zip(bufferSizeArray, range(len(bufferSizeArray))):
        
        Arithmetic_Mean = uploadProcess( 
                            minimal_framesize = a_small_minimal_framesize, 
                            estimatingType = "A.M.", 
                            pTrackUsed=trackUsed, 
                            pBufferTime = bufferTimeInArray/FPS)

        count_skip_AM = Arithmetic_Mean[2]
        AM_LossRateMatrix[ix][iy] = count_skip_AM/(howLongIsVideoInSeconds*FPS)
        AM_BitrateMatrix[ix][iy] = Arithmetic_Mean[0] / howLongIsVideoInSeconds 
        print("Arithmetic Mean. Bitrate :(M = "+str(trackUsed)+"): " + str(Arithmetic_Mean[0]/howLongIsVideoInSeconds) + 
            " (Mbps). Loss rate: " + str(count_skip_AM/(howLongIsVideoInSeconds*FPS)))

pyplot.xlabel("Initial buffer time (in 1/FPS seconds)")
pyplot.ylabel("Loss rate")
for ix in range(len(mAxis)):
    pyplot.plot(bufferSizeArray, AM_LossRateMatrix[ix], '-s', color=colorList[ix], markersize=2, linewidth=1)
pyplot.plot(bufferSizeArray, Minimal_Lossrate, '-s', color = "black", markersize = 2, linewidth = 1)
pyplot.plot(bufferSizeArray, Cond_Lossrate, '-s', color='blue',markersize=2, linewidth=1)
# pyplot.plot(bufferSizeArray, Marginal_Lossrate, '-s', color='purple',markersize=2, linewidth=1)
AMLegend = ["A.M. K=" + str(trackUsed) for trackUsed in mAxis]
pyplot.axhline(y=0.05, color='green', linestyle='-', linewidth=1)
pyplot.legend(AMLegend + 
            ["Fixed as Minimal"] +
            ["Empirical Condt'l"] + 
            # ["Marginal"] + 
            ["Const. 0.05"], 
            loc="best")
pyplot.tick_params(axis='x', labelsize=14)
pyplot.tick_params(axis='y', labelsize=14)
pyplot.show()

pyplot.xlabel("Initial buffer time (in 1/FPS seconds)")
pyplot.ylabel("Bitrate (in Mbps)")
for ix in range(len(mAxis)):
    pyplot.plot(bufferSizeArray, AM_BitrateMatrix[ix], '-s', color=colorList[ix], markersize=2, linewidth=1)
pyplot.plot(bufferSizeArray, Minimal_Bitrate, '-s', color = "black", markersize = 2, linewidth = 1)
pyplot.plot(bufferSizeArray, Cond_Bitrate, '-s', color='blue',markersize=2, linewidth=1)
# pyplot.plot(bufferSizeArray, Marginal_Bitrate, '-s', color='purple',markersize=2, linewidth=1)
AMLegend = ["A.M. K=" + str(trackUsed) for trackUsed in mAxis]
pyplot.legend(AMLegend + 
            ["Fixed as Minimal"] +
            ["Empirical Condt'l"], 
            # ["Marginal"],
            loc="best")
pyplot.tick_params(axis='x', labelsize=14)
pyplot.tick_params(axis='y', labelsize=14)
pyplot.show()


print("----------------------------------------------------------------------------------------")
print("----------- Now the explantory variable changes to be minimal frame size ---------------")
print("----------------------------------------------------------------------------------------")

# Lets do Bitrate and loss rate against minimal frame size
Cond_Lossrate_MFS = []
Cond_Bitrate_MFS = []
Minimal_Lossrate_MFS = []
Minimal_Bitrate_MFS = []
Marginal_Lossrate_MFS = []
Marginal_Bitrate_MFS = []

minFrameSizes = np.linspace(a_small_minimal_framesize, 0.15, num=3)

for thisMFS in minFrameSizes:
    ConditionalProposed_MFS = uploadProcess(
                        minimal_framesize= thisMFS, 
                        estimatingType = "ProbabilityPredict", 
                        pTrackUsed=5, 
                        pBufferTime = 1/FPS)

    count_skip_conditional_MFS = ConditionalProposed_MFS[2]
    Cond_Lossrate_MFS.append(count_skip_conditional_MFS/(howLongIsVideoInSeconds*FPS))
    Cond_Bitrate_MFS.append(ConditionalProposed_MFS[0]/howLongIsVideoInSeconds)
    print("Cond'l Proposed Method. Bitrate: " + str(ConditionalProposed_MFS[0]/howLongIsVideoInSeconds) + 
        " (Mbps). Loss rate: " + str(count_skip_conditional_MFS/(howLongIsVideoInSeconds*FPS)) )

    MinimalFrameScheme_MFS = uploadProcess(
                        minimal_framesize = thisMFS, 
                        estimatingType = "MinimalFrame", 
                        pTrackUsed=0, 
                        pBufferTime = 1/FPS)

    count_skip_minimal_MFS = MinimalFrameScheme_MFS[2]
    Minimal_Lossrate_MFS.append(count_skip_minimal_MFS/(howLongIsVideoInSeconds*FPS))
    Minimal_Bitrate_MFS.append(MinimalFrameScheme_MFS[0] / howLongIsVideoInSeconds )
    print("Minimal Framesize Method. Bitrate: " + str(MinimalFrameScheme_MFS[0] / howLongIsVideoInSeconds) + 
        " (Mbps). Loss rate: " + str(count_skip_minimal_MFS/(howLongIsVideoInSeconds*FPS)))

    MarginalScheme_MFS = uploadProcess(
                        minimal_framesize = thisMFS, 
                        estimatingType = "Marginal", 
                        pTrackUsed=5, 
                        pBufferTime = 1/FPS)

    count_skip_marginal_MFS = MarginalScheme_MFS[2]
    Marginal_Lossrate_MFS.append(count_skip_marginal_MFS/(howLongIsVideoInSeconds*FPS))
    Marginal_Bitrate_MFS.append(MarginalScheme_MFS[0] / howLongIsVideoInSeconds )
    print("Marginal Framesize Method. Bitrate: " + str(MarginalScheme_MFS[0] / howLongIsVideoInSeconds) + 
        " (Mbps). Loss rate: " + str(count_skip_marginal_MFS/(howLongIsVideoInSeconds*FPS)))

    print("-------------------------------------------------")


AM_LossRateMatrix_MFS = [ [0] * len(minFrameSizes)  for _ in range(len(mAxis))]
AM_BitrateMatrix_MFS  = [ [0] * len(minFrameSizes)  for _ in range(len(mAxis))]
for trackUsed, ix in zip(mAxis,range(len(mAxis))):
    for minFS, iy in zip(minFrameSizes, range(len(minFrameSizes))):
        
        Arithmetic_Mean_MFS = uploadProcess( 
                            minimal_framesize = minFS, 
                            estimatingType = "A.M.", 
                            pTrackUsed = trackUsed, 
                            pBufferTime = 1/FPS)

        count_skip_AM_MFS = Arithmetic_Mean_MFS[2]
        AM_LossRateMatrix_MFS[ix][iy] = count_skip_AM_MFS/(howLongIsVideoInSeconds*FPS)
        AM_BitrateMatrix_MFS[ix][iy] = Arithmetic_Mean_MFS[0] / howLongIsVideoInSeconds 
        print("Arithmetic Mean. Bitrate :(M = "+str(trackUsed)+"): " + str(Arithmetic_Mean_MFS[0]/howLongIsVideoInSeconds) + 
            " (Mbps). Loss rate: " + str(count_skip_AM_MFS/(howLongIsVideoInSeconds*FPS)))
            
pyplot.xlabel("Minimal frame size s_min (in Mbit)")
pyplot.ylabel("Loss rate")
for ix in range(len(mAxis)):
    pyplot.plot(minFrameSizes, AM_LossRateMatrix_MFS[ix], '-s', color=colorList[ix], markersize=2, linewidth=1)
pyplot.plot(minFrameSizes, Minimal_Lossrate_MFS, '-s', color = "black", markersize = 2, linewidth = 1)
pyplot.plot(minFrameSizes, Cond_Lossrate_MFS, '-s', color='blue',markersize=2, linewidth=1)
pyplot.plot(minFrameSizes, Marginal_Lossrate_MFS, '-s', color='purple',markersize=2, linewidth=1)
AMLegend = ["A.M. K=" + str(trackUsed) for trackUsed in mAxis]
pyplot.axhline(y=0.05, color='green', linestyle='-', linewidth=1)
pyplot.legend(AMLegend + 
            ["Fixed as Minimal"] +
            ["Empirical Condt'l"] + 
            ["Marginal"] + 
            ["Const. 0.05"], 
            loc="best")
pyplot.tick_params(axis='x', labelsize=14)
pyplot.tick_params(axis='y', labelsize=14)
pyplot.show()

pyplot.xlabel("Minimal frame size s_min (in Mbit)")
pyplot.ylabel("Bitrate (in Mbps)")
for ix in range(len(mAxis)):
    pyplot.plot(minFrameSizes, AM_BitrateMatrix_MFS[ix], '-s', color=colorList[ix], markersize=2, linewidth=1)
pyplot.plot(minFrameSizes, Minimal_Bitrate_MFS, '-s', color = "black", markersize = 2, linewidth = 1)
pyplot.plot(minFrameSizes, Cond_Bitrate_MFS, '-s', color='blue',markersize=2, linewidth=1)
pyplot.plot(minFrameSizes, Marginal_Bitrate_MFS, '-s', color='purple',markersize=2, linewidth=1)
AMLegend = ["A.M. K=" + str(trackUsed) for trackUsed in mAxis]
pyplot.legend(AMLegend + 
            ["Fixed as Minimal"] +
            ["Empirical Condt'l"] +
            ["Marginal"],
            loc="best")
pyplot.tick_params(axis='x', labelsize=14)
pyplot.tick_params(axis='y', labelsize=14)
pyplot.show()