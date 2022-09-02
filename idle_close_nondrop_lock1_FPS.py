# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     May 12, 2022


# Largest M means when doing the conditioning step, I only preserve the closet M
# And this provides the real throughput in the assigned time

import numpy as np
from numpy.core.fromnumeric import mean
import utils as utils
import matplotlib.pyplot as pyplot
from numpy import  quantile, var

network_trace_dir = './dataset/fyp_lab/'
howmany_Bs_IN_1Mb = 1024*1024/8


FPS = 60
whichVideo = 8

# Testing Set Size
howLongIsVideoInSeconds = 3020

networkEnvTime = [] 
networkEnvPacket= [] 
count = 0
initialTime = 0
packet_level_integral_C_training = []
packet_level_time_training = []


cut_off_time = 3000

assert cut_off_time < howLongIsVideoInSeconds

# In case that there are multiple packets in trace data having same timestamps, we merge them
for suffixNum in range(whichVideo,whichVideo+1):
    with open( network_trace_dir+ str(suffixNum) + ".txt" ) as traceDateFile:
        for eachLine in traceDateFile:
            parse = eachLine.split()
            if (count==0):
                initialTime = float(parse[0])
            nowFileTime = float(parse[0]) 
            if (len(networkEnvTime)>0 and nowFileTime - initialTime != networkEnvTime[-1]):
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
            elif (len(networkEnvTime)==0):
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
            else:
                networkEnvPacket[-1] += float(parse[1]) / howmany_Bs_IN_1Mb 
            count = count  +1 


# for suffixNum in range(whichVideo,whichVideo+1):
#     with open( network_trace_dir+ str(suffixNum) + ".txt" ) as traceDateFile:
#         for eachLine in traceDateFile:
#             parse = eachLine.split()
#             if (count==0):
#                 initialTime = (count+1) * 0.01
#             nowFileTime = float(parse[0]) 
#             if (len(networkEnvTime)>0 and nowFileTime - initialTime != networkEnvTime[-1]):
#                 networkEnvTime.append((count+1) * 0.01 - initialTime)
#                 networkEnvPacket.append( max(np.random.normal(1.5, 0.05, 1), 0.000000000001) ) 
#             elif (len(networkEnvTime)==0):
#                 networkEnvTime.append((count+1) * 0.01 - initialTime)
#                 networkEnvPacket.append(max(np.random.normal(1.5, 0.05, 1), 0.000000000001)) 
#             else:
#                 networkEnvPacket[-1] += max(np.random.normal( 1.5, 0.05, 1), 0.000000000001)
#             count = count  +1 


# Just have a quick idea of the mean throughput
throughputEstimateInit = sum(networkEnvPacket[0:10000]) / (networkEnvTime[10000-1]-networkEnvTime[0])

print("len of networkEnvPacket=" + str(len(networkEnvPacket)))

# pyplot.plot(networkEnvPacket[0:10000])
# pyplot.show()


print( str(throughputEstimateInit) + "Mbps, this is mean throughput")

# Mean calculation done.

pEpsilon = 0.05
M = 750

controlled_epsilon = pEpsilon





def uploadProcess( minimal_framesize, estimatingType, pTrackUsed, pBufferTime, sendingDummyData, dummyDataSize):
    
    frame_prepared_time = []

    throughputHistoryLog = []
    transmitHistoryTimeLog = []
    transmitHistoryTimeCum = []
    realVideoFrameSize = []

    # This is to SKIP the training part of the data.
    # Hence ensures that training data is not over-lapping with testing data
    # Mock up the frame generating times
    tempTimeTrack = 0
    runningTime = 0
    for _ in range( howLongIsVideoInSeconds * FPS ):
        frame_prepared_time.append(tempTimeTrack)
        tempTimeTrack = tempTimeTrack + 1/FPS
    
    uploadDuration = 0

    count_skip = 0

    sendDummyFrame = False

    thisFrameSize = 0

    videoCumsize = 0

    now_go_real = False

    count_Cond_AlgoTimes = 0
    countExceed = 0
    countExceeds = []
    cumPartSize = 0

    consecutive_skip = 0
    consecutive_skip_box = []

    countFrame = 0


    residual = []

    effectCount = 0
    failCount = 0

    # Note that the frame_prepared_time every time is NON-SKIPPABLE
    for singleFrame in range( howLongIsVideoInSeconds * FPS ):

        if singleFrame % 100 ==0:
            print("singleFrame=" + str(singleFrame)  + "   and runningTime is: " + str(runningTime) )

        if (runningTime >= cut_off_time):
            if (frame_prepared_time[singleFrame] < cut_off_time ):
                continue
            # print("frame_prepared_time[send 0] = " + str(frame_prepared_time[singleFrame]))
            now_go_real = True
            countFrame += 1
            # print("now running Time= " + str(runningTime))

        # totalNumberList[  min(floor(runningTime), len(totalNumberList)-1 ) ] += 1
        if ( 
            # singleFrame % (5 * FPS) == 0 and 
            estimatingType == "ProbabilityPredict"):
                if (now_go_real and singleFrame % 100 ==0):
                    print("Frame: " + str(singleFrame ) +" .Now is time: " + str(runningTime) 
                        + "--- Cond (with or w/o dummy) count times: " +str(count_Cond_AlgoTimes) + " ---part Size: " +str(cumPartSize) 
                        + "  Exceed counts: " + str(countExceed) + " exceed ratio: " + str(countExceed/100) + " effect count: " + str(effectCount) 
                    )
                    countExceeds.append(countExceed/100)
                    count_Cond_AlgoTimes = 0
                    cumPartSize = 0
                    countExceed = 0

        ########################################################################################

        if (runningTime > howLongIsVideoInSeconds):
            break 


        # Need to wait for a new frame
        if (singleFrame >0 and  runningTime < frame_prepared_time[singleFrame]):
            # Then we need to wait until singleframe is generated and available to send.
            if (sendingDummyData == False):
                # if does not need to send dummy, then wait until frame is ready without doing anything
                runningTime = frame_prepared_time[singleFrame]
            elif (sendingDummyData == True):
                sendDummyFrame = True


        countLoop = 0
        # Send dummy frame if necessary
        if (sendDummyFrame == True):   
            while (singleFrame >0 and  runningTime < frame_prepared_time[singleFrame]):
                countLoop += 1
                thisFrameSize =  dummyDataSize
                uploadFinishTime = utils.paper_frame_upload_finish_time( 
                                        runningTime= runningTime,
                                        packet_level_data= networkEnvPacket,
                                        packet_level_timestamp= networkEnvTime,
                                        framesize= thisFrameSize)[0]
                # We record the sent frames' information in this array.
                if (uploadFinishTime<=howLongIsVideoInSeconds):
                    realVideoFrameSize.append(thisFrameSize)

                uploadDuration = uploadFinishTime - runningTime
                runningTime = runningTime + uploadDuration 
                # print(str(singleFrame)+ "  uploadDuration: " +str(uploadDuration))

                throughputMeasure =  thisFrameSize / uploadDuration
                throughputHistoryLog.append(throughputMeasure)
                transmitHistoryTimeLog.append(uploadDuration)

                if (len(transmitHistoryTimeCum)>0):
                    transmitHistoryTimeCum.append(transmitHistoryTimeCum[-1]+uploadDuration)
                else:
                    transmitHistoryTimeCum.append(uploadDuration)
        
        if (countLoop!=0):
            print("countLoop = " + str(countLoop))
        sendDummyFrame == False

        
        #######################################################################################
        suggestedFrameSize = -np.Infinity

        if (consecutive_skip >=1):
            consecutive_skip_box.append(consecutive_skip)
        consecutive_skip = 0

        T_i = 1/FPS

        switch_to_AM = False

        if (estimatingType == "ProbabilityPredict"):
            backLen = FPS * 300
            timeSlot= T_i

            if (runningTime >= cut_off_time):
                lookbackwardHistogramS =  utils.generatingBackwardSizeFromLog_fixLen(
                                            pastDurations= transmitHistoryTimeLog,
                                            pastDurationsCum= transmitHistoryTimeCum,
                                            pastSizes= realVideoFrameSize, 
                                            backLen= backLen,
                                            timeSlot= timeSlot
                                        )

            else:
                lookbackwardHistogramS = []
            
            
            if (len(lookbackwardHistogramS)>0):
                # print("已經用了")
                # conditional
                Shat_iMinus1 = lookbackwardHistogramS[-1]
                need_index = utils.extract_nearest_M_values_index(lookbackwardHistogramS, Shat_iMinus1, M )
                need_index = np.array(need_index)
                need_index_plus1 = need_index + 1
                decision_list = [lookbackwardHistogramS[a] for a in need_index_plus1 if a < len(lookbackwardHistogramS)]

                # marginal
                # decision_list = lookbackwardHistogramS
                

                # P controller?
                # if effectCount > 100:
                #     controlled_epsilon = (pEpsilon - failCount/effectCount) * 0.1 + controlled_epsilon
                # else:
                #     controlled_epsilon = pEpsilon
                
                # controlled_epsilon = min(controlled_epsilon, 0.07)
                # controlled_epsilon = max(controlled_epsilon, 0.03)
                
                suggestedFrameSize = quantile(decision_list, controlled_epsilon)
                count_Cond_AlgoTimes += 1
                cumPartSize += suggestedFrameSize

                if ( runningTime > cut_off_time):
                    maxData = utils.calMaxData(prevTime=runningTime, 
                                        laterTime=runningTime+timeSlot, 
                                        packet_level_timestamp= networkEnvTime,
                                        packet_level_data= networkEnvPacket,)

                    # print("maxData is: " + str(maxData))

                    # true.append(1)
                    residual.append( (mean(decision_list) - maxData)/timeSlot )
                    effectCount += 1

                    if (suggestedFrameSize > maxData):
                        countExceed += 1
                        failCount += 1

                    # pyplot.hist(decision_list, bins=50)
                    
                    # pyplot.axvline(x= maxData, color="red")
                    # pyplot.axvline(x=mean(decision_list), color="gold")
                    # pyplot.axvline(x=suggestedFrameSize, color="green")
                    # # pyplot.axvline(x = mean(denoised_quantile.confidence_interval), color="violet")
                    # pyplot.legend(["Max. throughput s.t. No Drop",
                    #              "(Unbiased) Estimated", 
                    #              "Suggested (Aka. chosen)", 
                    #             #  "Bootstrap value"
                    #              ])
                    # pyplot.ylabel("number of occurrences in the past")
                    # pyplot.xlabel("size (in Mb)")
                    # pyplot.title("Estimating distribution of frame No." + str(singleFrame) + "'s size")
                    # pyplot.show()

            elif (len(throughputHistoryLog)==0 or len(lookbackwardHistogramS) == 0):
                switch_to_AM = True


        if ( (estimatingType == "A.M." or switch_to_AM == True ) and len(throughputHistoryLog) > 0 ):
            adjustedAM_Nume = sum(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)])
            adjustedAM_Deno = [ a/b for a,b in zip(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)], 
                                        throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog)]) ]
            C_i_hat_AM = adjustedAM_Nume/sum(adjustedAM_Deno)
            suggestedFrameSize = (1/FPS) * C_i_hat_AM

        elif (estimatingType == "MinimalFrame"):
            suggestedFrameSize = minimal_framesize


        # Above is case-wise, now it is general to transmit a (video) frame
        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize )
    
        # Until now, the suggestedFrameSize is fixed.
        #######################################################################################
        uploadFinishTime = utils.paper_frame_upload_finish_time( 
                                runningTime= runningTime,
                                packet_level_data= networkEnvPacket,
                                packet_level_timestamp= networkEnvTime,
                                framesize= thisFrameSize)[0]


        # We record the sent frames' information in this array.
        # if (uploadFinishTime<=howLongIsVideoInSeconds):
        realVideoFrameSize.append(thisFrameSize)

        uploadDuration = uploadFinishTime - runningTime
        runningTime = runningTime + uploadDuration 

                # Encounter with frame dropping!!!
        if (uploadDuration >= 1/FPS):
            # print("Some frame skipped!")
            if (now_go_real):
                count_skip = count_skip + 1
                # print("xxxxxx")
                consecutive_skip += 1


        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)
        transmitHistoryTimeLog.append(uploadDuration)
        if (len(transmitHistoryTimeCum)>0):
            transmitHistoryTimeCum.append(transmitHistoryTimeCum[-1]+uploadDuration)
        else:
            transmitHistoryTimeCum.append(uploadDuration)
        
        if (now_go_real):
            videoCumsize += thisFrameSize

        # print("It's frame No." + str(singleFrame) + ". And the time cost is" + str(uploadDuration) + ". And size is " +str(thisFrameSize) + "Mb")



    if (len(consecutive_skip_box)>0):
        print(str(mean(consecutive_skip_box)) + " " +str(max(consecutive_skip_box))   )



    # pyplot.plot(true)
    print("effectCount is: " + str(effectCount) + " failCount: " + str(failCount) )
    print("Fail rate among effective ones: "+ str(failCount/effectCount))
    print("Mean of residual: " + str(mean(residual)) + " Variance of residual: "+ str(var(residual)))
    pyplot.plot(residual)
    pyplot.xlabel("frame No.")
    pyplot.ylabel("Residual of true throughput and estimated throughput")
    pyplot.title("Estimated - true value throughput")
    pyplot.show()


    per100lr = countExceeds[1:]
    print("Mean of per100lr: " + str( mean(per100lr) ) + ", variance of per100lr: " + str(var(per100lr)))
    pyplot.plot(per100lr, color="blue")
    pyplot.xlabel("100 frames per slot")
    pyplot.ylabel("loss rate of that 100 frames")
    pyplot.axhline(0.05, color="red")
    pyplot.legend(["real loss rate", "target 0.05"])
    pyplot.show()

    return [videoCumsize, [], count_skip, minimal_framesize, dummyDataSize, countFrame]




colorList = ["red", "orange", "goldenrod"]
# bufferSizeArray = np.arange(0, 6.25, step = 2)
Cond_Lossrate = []
Cond_Bitrate = []
Minimal_Lossrate = []
Minimal_Bitrate = []
Marginal_Lossrate = []
Marginal_Bitrate = []

a_small_minimal_framesize = 0.00000005
mAxis = [5,16,128]

# Lets do Bitrate and loss rate against minimal frame size
Cond_Lossrate_MFS = []
Cond_Bitrate_MFS = []

minFrameSizes = np.linspace(a_small_minimal_framesize, 0.8 , num=3)
dummySizes = np.linspace(0.01*1000/1024, 0.1*1000/1024, num=2)
# dummySizes = [ 0.05*1000/1024 ]
Cond_Lossrate_Dummy_MFS = [ [0] * len(minFrameSizes)  for _ in range(len(dummySizes))]
Cond_Bitrate_Dummy_MFS =  [ [0] * len(minFrameSizes)  for _ in range(len(dummySizes))]
Minimal_Lossrate_MFS = []
Minimal_Bitrate_MFS = []
Marginal_Lossrate_MFS = []
Marginal_Bitrate_MFS = []


# some_initial_buffer = 1/FPS
some_initial_buffer = 0



for thisMFS, idxMFS in zip(minFrameSizes, range(len(minFrameSizes))):
    # ConditionalProposed_MFS = uploadProcess(
    #                     minimal_framesize= thisMFS, 
    #                     estimatingType = "ProbabilityPredict", 
    #                     pTrackUsed=5, 
    #                     pBufferTime = some_initial_buffer,
    #                     sendingDummyData= False,
    #                     dummyDataSize= 0 )

    # count_skip_conditional_MFS = ConditionalProposed_MFS[2]
    # Cond_Lossrate_MFS.append(count_skip_conditional_MFS/((howLongIsVideoInSeconds-cut_off_time)*FPS))
    # Cond_Bitrate_MFS.append(ConditionalProposed_MFS[0]/(howLongIsVideoInSeconds-cut_off_time))
    # print("Cond'l Proposed Method. Bitrate: " + str(ConditionalProposed_MFS[0]/(howLongIsVideoInSeconds-cut_off_time)) + 
    #     " (Mbps). Loss rate: " + str(count_skip_conditional_MFS/((howLongIsVideoInSeconds-cut_off_time)*FPS)) )




    for dummySize, idx in zip(dummySizes,range(len(dummySizes))):
        ConditionalProposed_MFS_Dummy = uploadProcess(
                            minimal_framesize= thisMFS, 
                            estimatingType = "ProbabilityPredict", 
                            pTrackUsed=5, 
                            pBufferTime = some_initial_buffer,
                            sendingDummyData= True, 
                            dummyDataSize= dummySize)

        count_skip_conditional_MFS_dummy = ConditionalProposed_MFS_Dummy[2]
        Cond_Lossrate_Dummy_MFS[idx][idxMFS]= (count_skip_conditional_MFS_dummy/ConditionalProposed_MFS_Dummy[-1])
        Cond_Bitrate_Dummy_MFS[idx][idxMFS]= (ConditionalProposed_MFS_Dummy[0]/(howLongIsVideoInSeconds-cut_off_time))
        print("Cond'l Proposed Method (dummysize=" + str(dummySize*1024/8) + "KB). Bitrate: " + str(Cond_Bitrate_Dummy_MFS[idx][idxMFS]) + 
            " (Mbps). Loss rate: " + str(count_skip_conditional_MFS_dummy/ConditionalProposed_MFS_Dummy[-1]) )




    MinimalFrameScheme_MFS = uploadProcess(
                        minimal_framesize = thisMFS, 
                        estimatingType = "MinimalFrame", 
                        pTrackUsed=0, 
                        pBufferTime = some_initial_buffer,
                        sendingDummyData= False,
                        dummyDataSize= 0 )

    count_skip_minimal_MFS = MinimalFrameScheme_MFS[2]
    Minimal_Lossrate_MFS.append(count_skip_minimal_MFS/((howLongIsVideoInSeconds-cut_off_time)*FPS))
    Minimal_Bitrate_MFS.append(MinimalFrameScheme_MFS[0] / (howLongIsVideoInSeconds-cut_off_time) )
    print("Minimal Framesize Method. Bitrate: " + str(MinimalFrameScheme_MFS[0] / (howLongIsVideoInSeconds-cut_off_time)) + 
        " (Mbps). Loss rate: " + str(count_skip_minimal_MFS/((howLongIsVideoInSeconds-cut_off_time)*FPS)))



    # MarginalScheme_MFS = uploadProcess(
    #                     minimal_framesize = thisMFS, 
    #                     estimatingType = "Marginal", 
    #                     pTrackUsed=5, 
    #                     pBufferTime = some_initial_buffer, 
    #                     sendingDummyData= False, 
    #                     dummyDataSize= 0)

    # count_skip_marginal_MFS = MarginalScheme_MFS[2]
    # Marginal_Lossrate_MFS.append(count_skip_marginal_MFS/(howLongIsVideoInSeconds*FPS))
    # Marginal_Bitrate_MFS.append(MarginalScheme_MFS[0] / howLongIsVideoInSeconds )
    # print("Marginal Framesize Method. Bitrate: " + str(MarginalScheme_MFS[0] / howLongIsVideoInSeconds) + 
    #     " (Mbps). Loss rate: " + str(count_skip_marginal_MFS/(howLongIsVideoInSeconds*FPS)))

    print("-------------------------------------------------")


AM_LossRateMatrix_MFS = [ [0] * len(minFrameSizes)  for _ in range(len(mAxis))]
AM_BitrateMatrix_MFS  = [ [0] * len(minFrameSizes)  for _ in range(len(mAxis))]
for trackUsed, ix in zip(mAxis,range(len(mAxis))):
    for minFS, iy in zip(minFrameSizes, range(len(minFrameSizes))):
        
        Arithmetic_Mean_MFS = uploadProcess( 
                            minimal_framesize = minFS, 
                            estimatingType = "A.M.", 
                            pTrackUsed = trackUsed, 
                            pBufferTime = some_initial_buffer,
                            sendingDummyData= False, 
                            dummyDataSize= 0 )

        count_skip_AM_MFS = Arithmetic_Mean_MFS[2]
        AM_LossRateMatrix_MFS[ix][iy] = count_skip_AM_MFS/((howLongIsVideoInSeconds-cut_off_time)*FPS)
        AM_BitrateMatrix_MFS[ix][iy] = Arithmetic_Mean_MFS[0] / (howLongIsVideoInSeconds-cut_off_time) 
        print("Arithmetic Mean. Bitrate :(M = "+str(trackUsed)+"): " + str(Arithmetic_Mean_MFS[0]/(howLongIsVideoInSeconds-cut_off_time)) + 
            " (Mbps). Loss rate: " + str(count_skip_AM_MFS/((howLongIsVideoInSeconds-cut_off_time)*FPS)))




pyplot.xlabel("Minimal frame size s_min (in Mbit)")
pyplot.ylabel("Loss rate")
# for ix in range(len(mAxis)):
#     pyplot.plot(minFrameSizes, AM_LossRateMatrix_MFS[ix], '-s', markersize=2, linewidth=1)

# pyplot.plot(minFrameSizes, Minimal_Lossrate_MFS, '-s', color = "black", markersize = 2, linewidth = 1)


# pyplot.plot(minFrameSizes, Cond_Lossrate_MFS, '-s',markersize=2, linewidth=1.5)
for idx in range(len(dummySizes)):
    pyplot.plot(minFrameSizes, Cond_Lossrate_Dummy_MFS[idx], '-s', markersize=4, linewidth=2)
# pyplot.plot(minFrameSizes, Marginal_Lossrate_MFS, '-s', markersize=2, linewidth=1.5)

# AMLegend = ["A.M. K=" + str(trackUsed) for trackUsed in mAxis]
DummyLegend = ["Dummy Cond " + str( "{:.2f}".format(dummySize * 1024/8) ) + "KB" for dummySize in dummySizes] 
pyplot.axhline(y=0.05, linestyle='-', linewidth=1)
pyplot.legend(
            # AMLegend + 
            # ["Fixed as Minimal"] +
            # ["Empirical Condt'l"] + 
            DummyLegend +
            # ["Marginal"] + 
            ["Const. 0.05"], 
            loc="best")
pyplot.tick_params(axis='x', labelsize=14)
pyplot.tick_params(axis='y', labelsize=14)
pyplot.show()





pyplot.xlabel("Minimal frame size s_min (in Mbit) FPS="+str(FPS) )
pyplot.ylabel("Bitrate (in Mbps)")
# for ix in range(len(mAxis)):
#     pyplot.plot(minFrameSizes, AM_BitrateMatrix_MFS[ix], '-s', markersize=2, linewidth=1)

# pyplot.plot(minFrameSizes, Minimal_Bitrate_MFS, '-s', color = "black", markersize = 2, linewidth = 1)

# pyplot.plot(minFrameSizes, Cond_Bitrate_MFS, '-s', markersize=2, linewidth=1.5)
for idx in range(len(dummySizes)):
    pyplot.plot(minFrameSizes, Cond_Bitrate_Dummy_MFS[idx], '-s', markersize=4, linewidth=2)
# pyplot.plot(minFrameSizes, Marginal_Bitrate_MFS, '-s',markersize=2, linewidth=1.5)

pyplot.legend(
            # AMLegend + 
            # ["Fixed as Minimal"] + 
            # ["Empirical Condt'l"] +
            DummyLegend, 
            # ["Marginal"],
            loc="best")
pyplot.tick_params(axis='x', labelsize=14)
pyplot.tick_params(axis='y', labelsize=14)
pyplot.show()