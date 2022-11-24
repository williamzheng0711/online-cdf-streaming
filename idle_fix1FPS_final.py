# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     May 12, 2022


# Largest M means when doing the conditioning step, I only preserve the closet M
# And this provides the real throughput in the assigned time

from statsmodels.graphics.tsaplots import *
import numpy as np
from numpy.core.fromnumeric import mean
import utils as utils
import matplotlib.pyplot as pyplot
from numpy import  quantile, var



# The following are GLOBAL variables
howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024
FPS = 30                         # frame per second
whichVideo = 11                  # No. of trace data we perfrom a simulation on
cut_off_time1 = 200              # This time is for accumulate the PDF space
cut_off_time2 = 60                # to accumulate the percentile
howLongIsVideoInSeconds = cut_off_time1 + cut_off_time2 + 300   # terminate simulation at such time
pEpsilon = 0.05
controlled_epsilon = pEpsilon
M = 70
backSeconds = 60

assert cut_off_time1 + cut_off_time2 < howLongIsVideoInSeconds

# Zhang Yuming Showing: "In case that there are multiple packets in trace data having same timestamps, we merge them."
traceDir = './dataset/fyp_lab/'
count = 0
initialTime = 0
networkEnvTime = [] 
networkEnvPacket= [] 
for sufNum in range(whichVideo, whichVideo+1):
    with open( traceDir + str(sufNum) + ".txt" ) as traceDataFile:
        for line in traceDataFile:
            parse = line.split()
            if (count==0):
                initialTime = float(parse[0])
            fileTime = float(parse[0]) 
            if (len(networkEnvTime)>0 and fileTime - initialTime != networkEnvTime[-1]): # common cases
                networkEnvTime.append(fileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
            elif (len(networkEnvTime)==0):
                networkEnvTime.append(fileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
            else: # deal with packets with the same timestamp
                networkEnvPacket[-1] += float(parse[1]) / howmany_Bs_IN_1Mb 
            count = count  +1 

throughputEstimateInit = sum(networkEnvPacket[0:10000]) / (networkEnvTime[10000-1]-networkEnvTime[0]) # Just have a quick glance of the mean throughput
print( str(throughputEstimateInit) + "Mbps, this is mean throughput")



def uploadProcess( minimal_framesize, estimatingType, pTrackUsed, pBufferTime, sendingDummyData, subDummySize):
    
    frame_prepared_times = []       
    tempTimeTrack = 0
    for _ in range( howLongIsVideoInSeconds * FPS ):
        tempTimeTrack = tempTimeTrack + 1/FPS
        frame_prepared_times.append(tempTimeTrack) 
    
    # variables that would be re-initialized every 100 frames
    count_Cond_AlgoTimes = 0        # Integer. A counter cannot exceed 100, because we see statistics every 100 frames
    countExceed = 0                 # Integer. No. of exceeds in the this 100 frames
    cumPartSize = 0                 # Float, calculating cummulative video (non dummy) size for this 100 frames.

    # variables that would get updated for all simulation
    runningTime = 0                 # "runningTime" is the simulation clock
    uploadDuration = 0              # transmission time for ANY thing (dummy or non-dummy)
    sendDummyFrame = False          # Boolean deciding whether to send dummy between any two frames 
    videoCumsize = 0                # Float restores cummulative video (not dummy) frame sizes AFTER cut_off_time
    now_go_real = False             # Boolean of whether simulation clock >= cut_off_time 
    effectCount = 0                 # Integer. Counts how many times our algorithm is applied on
    failCount = 0                   # Integer. Counting how many exceed during the whole simulation
    frameCntAfterCutoff = 0         # Integer. Counting how many frames after cut_off_time
    throughputHistoryLog = []       # Float array. Storing the avg. throughputs of every transmission (dummy and non-dummy)
    realVideoFrameSize = []         # Float array. Storing sizes of every transmission (dummy and non-dummy)
    transmitHistoryTimeLog = []     # Float array. Storing every transmission time (dummy and non-dummy)
    transmitHistoryTimeCum = []     # Float array. CumSum of "transmitHistoryTimeLog"
    exceedsRatios = []              # Float array. The i-th element is the loss ratio about the [100*i : 100*(i+1)]-th frames after cut-off time

    percentiles = []


    # Each elements in "frame_prepared_times" is NON-SKIPPABLE
    for singleFrame in range( howLongIsVideoInSeconds * FPS ):

        if (runningTime >= cut_off_time1 + cut_off_time2): # We want to know if we reached cut_off_time
            if (frame_prepared_times[singleFrame] < cut_off_time1 + cut_off_time2 ):
                continue
            now_go_real = True # Switch on this variable
            frameCntAfterCutoff += 1
        
        if (estimatingType == "ProbabilityPredict"):  
            if (now_go_real and singleFrame % 100 == 99):
                print("Frame: " + str(singleFrame) +" .Now is time: " + str(runningTime) 
                    + "--- Cond (with or w/o dummy) ---part Size: " +str(cumPartSize) 
                    + "  Exceed counts: " + str(countExceed) + " exceed ratio: " + str(countExceed/100) + " effect count: " + str(effectCount)  )
                exceedsRatios.append(countExceed/100)
                print(controlled_epsilon)
                count_Cond_AlgoTimes = 0; cumPartSize = 0; countExceed = 0


        if (runningTime > howLongIsVideoInSeconds): break    # now end the simulation as has reached the terminal


        ########################################################################################
        # Transmission of Dummy Part Starts Here. 

        # Need to wait for a new frame (we wait by send dummy)
        if (singleFrame >0 and  runningTime < frame_prepared_times[singleFrame]):  # wait until singleframe is available to transmit.
            if (sendingDummyData == False): # Don't need to send dummy. Just wait until frame is ready.
                runningTime = frame_prepared_times[singleFrame]
            elif (sendingDummyData == True):
                sendDummyFrame = True

        cntSubDummy = 0
        if (sendDummyFrame == True):  
            while (runningTime<frame_prepared_times[singleFrame] and singleFrame>0):
                cntSubDummy += 1 # count how many dummys in this row
                uploadFinishTime = utils.paper_frame_upload_finish_time(runningTime = runningTime,  # returns the finish time of transmitting this dummy
                                                                        packet_level_data = networkEnvPacket, 
                                                                        packet_level_timestamp = networkEnvTime,
                                                                        framesize = subDummySize,)[0] 

                # update simulation clock
                uploadDuration = uploadFinishTime - runningTime; 
                runningTime = uploadFinishTime 
                throughputMeasure =  subDummySize / uploadDuration
                
                # update the size logs
                throughputHistoryLog.append(throughputMeasure); 
                transmitHistoryTimeLog.append(uploadDuration)
                realVideoFrameSize.append(subDummySize)

                # update the time logs
                if (len(transmitHistoryTimeCum)>0):
                    transmitHistoryTimeCum.append(transmitHistoryTimeCum[-1]+uploadDuration)
                else:
                    transmitHistoryTimeCum.append(uploadDuration)
            # reach the out of while loop. Aka, we reached the time to send a non-dummy frame.
        sendDummyFrame == False

        # Transmission of Dummy Part Ends Here. 
        #######################################################################################


        ########################################################################################
        # Determination of "thisFrameSize" of Non-dummy Frame Part Starts Here. 

        suggestedFrameSize = -np.Infinity 

        T_i = 1/FPS # time allocation for transmission of a frame
        switch_to_AM = False 

        if (estimatingType == "ProbabilityPredict"):
            backLen = FPS * backSeconds
            timeSlot= T_i

            if (runningTime >= cut_off_time1):
                lookbackwardHistogramS =  utils.generatingBackwardSizeFromLog_fixLen(
                                            pastDurations= transmitHistoryTimeLog,
                                            pastDurationsCum= transmitHistoryTimeCum,
                                            pastSizes= realVideoFrameSize, 
                                            backLen= backLen,
                                            timeSlot= timeSlot, )

            else: lookbackwardHistogramS = []
            
            
            if (len(lookbackwardHistogramS)>0):
                # plot_pacf(x=lookbackwardHistogramS, lags=15, method="ywm")
                # pyplot.show()
                # plot_acf(x=np.array(lookbackwardHistogramS))
                # pyplot.show()
                # pyplot.plot(np.log(np.array(lookbackwardHistogramS)))
                # pyplot.show()
                loglookbackwardHistogramS = np.log(np.array(lookbackwardHistogramS))
                arg = np.argmax(pacf(np.array(loglookbackwardHistogramS))[1:])
                arg = arg + 1
                # if arg != 1: 
                #     plot_pacf(x=loglookbackwardHistogramS, lags=15, method="ywm")
                #     pyplot.show()
                if(runningTime >= cut_off_time1 + cut_off_time2): effectCount += 1
                # print("len of lookbackwardHistogramS: " + str(len(lookbackwardHistogramS)))
                Shat_iMinus1 = loglookbackwardHistogramS[-1*arg]
                need_index = utils.extract_nearest_M_values_index(loglookbackwardHistogramS, Shat_iMinus1, M)
                need_index = np.array(need_index)
                need_index_plus_arg = need_index + arg
                decision_list = [loglookbackwardHistogramS[a] for a in need_index_plus_arg if a < len(loglookbackwardHistogramS)]

                # whether to use P controller?
                # if effectCount > 100:
                #     controlled_epsilon = (pEpsilon - failCount/effectCount) * 0.1 + controlled_epsilon
                # else:
                #     controlled_epsilon = pEpsilon
                
                # controlled_epsilon = min(controlled_epsilon, 0.09)
                # controlled_epsilon = max(controlled_epsilon, 0.01)
                
                # pyplot.hist(percentiles, bins=50, cumulative=True, density=True)
                # pyplot.axline((0, 0), slope=1)
                # pyplot.show(block=False)
                # pyplot.pause(0.01)
                
                if (runningTime >= cut_off_time1 + cut_off_time2 and len(percentiles)>=cut_off_time2*FPS):
                    controlled_epsilon = np.quantile(percentiles, pEpsilon, method="median_unbiased")
                    # controlled_epsilon = pEpsilon
                    # print(controlled_epsilon)
                else:
                    controlled_epsilon = pEpsilon
                suggestedFrameSize = np.exp(quantile(decision_list, controlled_epsilon, method='median_unbiased'))
                count_Cond_AlgoTimes += 1
                if (runningTime >= cut_off_time1 + cut_off_time2): cumPartSize += suggestedFrameSize

                maxData = utils.calMaxData(prevTime=runningTime, 
                                        laterTime=runningTime+timeSlot, 
                                        packet_level_timestamp= networkEnvTime,
                                        packet_level_data= networkEnvPacket,)
                
                log_maxData = np.log(maxData)
                percentiles.append( np.count_nonzero(decision_list <= log_maxData) / len(decision_list) )
                percentiles = percentiles[max(len(percentiles)-cut_off_time2 * FPS, 0) : ]

                if ( runningTime > cut_off_time1 + cut_off_time2 and singleFrame > howLongIsVideoInSeconds * FPS ):
                    pyplot.hist(decision_list, bins=50)
                    pyplot.axvline(x= np.log(maxData), color="red")
                    # pyplot.axvline(x= minimal_framesize, color="black")
                    pyplot.axvline(x=mean(decision_list), color="gold")
                    pyplot.axvline(x=np.log(suggestedFrameSize), color="green")
                    # pyplot.axvline(x = mean(denoised_quantile.confidence_interval), color="violet")
                    pyplot.legend([
                                 "Max. throughput s.t. No Drop",
                                #  "Minimal frame size",
                                 "(Unbiased) Estimated", 
                                 "Suggested (Aka. chosen)", 
                                #  "Bootstrap value"
                                 ])
                    pyplot.ylabel("number of occurrences in the past")
                    pyplot.xlabel("size (in Mb)")
                    pyplot.title("Estimating distribution of frame No." + str(singleFrame) + "'s size")
                    pyplot.show()

            elif (len(throughputHistoryLog)==0 or len(lookbackwardHistogramS) == 0): 
                # Remember: when runningTime < cut_off_time, then assign len(lookbackwardHistogramS) = 0
                switch_to_AM = True

        if ((estimatingType == "A.M." or switch_to_AM == True) and len(throughputHistoryLog) > 0):
            adjustedAM_Nume = sum(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)])
            adjustedAM_Deno = [ a/b for a,b in zip(realVideoFrameSize[ max(0,len(realVideoFrameSize)-pTrackUsed,): len(realVideoFrameSize)], 
                                        throughputHistoryLog[ max(0,len(throughputHistoryLog)-pTrackUsed,): len(throughputHistoryLog)]) ]
            C_i_hat_AM = adjustedAM_Nume/sum(adjustedAM_Deno) # C denotes throughput, aka, size/time
            suggestedFrameSize = (1/FPS) * C_i_hat_AM

        elif (estimatingType == "MinimalFrame"):
            suggestedFrameSize = minimal_framesize

        thisFrameSize =  max ( suggestedFrameSize, minimal_framesize ) # above is case-wise, now it is general to transmit a (video) frame
    
        # Determination of "thisFrameSize" of Non-dummy Frame Part Ends Here. 
        #######################################################################################


        ########################################################################################
        # Transmission of the Non-dummy Frame (whose size is determined above) Starts Here. 
        uploadFinishTime = utils.paper_frame_upload_finish_time(runningTime= runningTime, packet_level_data= networkEnvPacket,
                                                                packet_level_timestamp= networkEnvTime, framesize= thisFrameSize)[0]
        uploadDuration = uploadFinishTime - runningTime; 
        runningTime = uploadFinishTime

        if (uploadDuration > 1/FPS):    # encounter with frame dropping
            if (now_go_real):
                countExceed += 1            # countExceed retores the No. of skips in the 100 frames
                failCount   += 1            # failCount restores the whole No. of skips starting from cut_off_time

        throughputMeasure =  thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)
        transmitHistoryTimeLog.append(uploadDuration)
        realVideoFrameSize.append(thisFrameSize)

        if (len(transmitHistoryTimeCum)>0):
            transmitHistoryTimeCum.append(transmitHistoryTimeCum[-1]+uploadDuration)
        else:
            transmitHistoryTimeCum.append(uploadDuration)
        
        if (now_go_real):
            videoCumsize += thisFrameSize
        # Transmission of the Non-dummy Frame (whose size is determined above) Ends Here.
        ########################################################################################

        # will go to next "singleFrame"


    per100lr = exceedsRatios[1:]
    print("Mean of per100lr: " + str( mean(per100lr) ) + ", variance of per100lr: " + str(var(per100lr)))
    pyplot.plot(per100lr, color="blue")
    pyplot.xlabel("100 frames per slot")
    pyplot.ylabel("loss rate of that 100 frames")
    pyplot.axhline(0.05, color="red")
    pyplot.legend(["real loss rate", "target 0.05"])
    pyplot.show()

    pyplot.hist(percentiles, bins=50, cumulative=True, density=True)
    pyplot.axline((0, 0), slope=1)
    pyplot.show()

    return [videoCumsize, [], failCount , minimal_framesize, subDummySize, frameCntAfterCutoff]



someMinimalFramesize = 0.005*1000/1024
someSubDummySize     = 0.005*1000/1024
someInitialBuffer    = 0                 # cannot go with buffer now

ConditionalProposed_MFS_Dummy = uploadProcess(minimal_framesize= someMinimalFramesize, 
                                                    estimatingType = "ProbabilityPredict", 
                                                    pTrackUsed = 100, 
                                                    pBufferTime = someInitialBuffer,
                                                    sendingDummyData= True, 
                                                    subDummySize= someSubDummySize )
