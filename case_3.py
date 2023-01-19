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
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX



# The following are GLOBAL variables
howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024
FPS = 30                         # frame per second
whichVideo = 17                  # No. of trace data we perfrom a simulation on
cut_off_time1 = 200              # This time is for accumulate the PDF space
cut_off_time2 = 60                # to accumulate the percentile
howLongIsVideoInSeconds = cut_off_time1 + cut_off_time2 +600   # terminate simulation at such time
pEpsilon = 0.05
controlled_epsilon = pEpsilon
M = 70
backSeconds = 60

assert cut_off_time1 + cut_off_time2 < howLongIsVideoInSeconds

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



def uploadProcess( minimal_framesize, estimatingType, pTrackUsed, pBufferTime):
    
    frame_prepared_times = []       
    tempTimeTrack = 0
    for _ in range( howLongIsVideoInSeconds * FPS ):
        tempTimeTrack = tempTimeTrack + 1/FPS
        frame_prepared_times.append(tempTimeTrack) 
    
    # variables that would be re-initialized every 100 frames
    count_Cond_AlgoTimes = 0        # Integer. A counter cannot exceed 100, because we see statistics every 100 frames
    countExceed = 0                 # Integer. No. of exceeds in the this 100 frames
    localNoneffectCount = 0            # Integer. No. of frames our algo does not applied on (directly dropped because of previous problematic frames)
    cumPartSize = 0                 # Float, calculating cummulative video (non dummy) size for this 100 frames.

    # variables that would get updated for all simulation
    runningTime = 0                 # "runningTime" is the simulation clock
    uploadDuration = 0              # transmission time for ANY thing (dummy or non-dummy)
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
    decision_list = []
    percentiles = []
    # bufferTime = pBufferTime

    startingFrame = -1


    for singleFrame in range( howLongIsVideoInSeconds * FPS ):

        if ( sum(transmitHistoryTimeLog) >= cut_off_time1 + cut_off_time2): # We want to know if we learned enough time
            frameCntAfterCutoff += 1

            if (frame_prepared_times[singleFrame] < cut_off_time1 + cut_off_time2 ):
                continue
            
            now_go_real = True # Switch on this variable
            if startingFrame == -1:
                startingFrame = singleFrame
                print("startingFrame is "  + str(startingFrame))
        
        # Renew some statistics
        if (estimatingType == "ProbabilityPredict"):  
            if (now_go_real and singleFrame % 100 == 99):
                if (localNoneffectCount!=100):
                    print("Frame: " + str(singleFrame) + " .Now is time: " + str(runningTime) 
                        + "  Exceed counts: " + str(countExceed) + " exceed ratio: " + str(countExceed/(100-localNoneffectCount)) + " effect count: " + str(effectCount) 
                        # + "  local Noneffect counts: " + str(localNoneffectCount) 
                        + "  Controlled epsilon: " + str(controlled_epsilon)) 
                    exceedsRatios.append(countExceed/(100-localNoneffectCount))
                    
                else: 
                    print("localNoneffectCount == 100, something strange happens")
                    exceedsRatios.append(-1)
                
                count_Cond_AlgoTimes = 0; 
                cumPartSize = 0; 
                countExceed = 0; 
                localNoneffectCount = 0

        # To return an end if needed
        if (runningTime > howLongIsVideoInSeconds): 
            print("endFrame is "  + str(singleFrame))
            break    # now end the simulation as has reached the terminal
        
        # If have spare time, we do nothing but wait for the next event
        if (singleFrame >0 and ( runningTime < frame_prepared_times[singleFrame])): 
            runningTime = frame_prepared_times[singleFrame]
        
        # We need to drop current frame.
        if runningTime > frame_prepared_times[singleFrame] + 1/FPS + pBufferTime:
            # The Python rounding scheme maybe is somehow not accurate... 
            # I define the difference need to be larger than 0.00001 to be significant
            if now_go_real and np.abs(frame_prepared_times[singleFrame] + 1/FPS + pBufferTime - runningTime) > 0.00001:
                print("有點問題" + str(runningTime) + ' = ' + str(runningTime) + " the last DDL =" + str(frame_prepared_times[singleFrame] + 1/FPS + pBufferTime))
                failCount += 1            # failCount restores the whole No. of skips starting from cut_off_time
                localNoneffectCount += 1
                # bufferTime = max(pBufferTime-max(runningTime-frame_prepared_times[singleFrame],0),0) 
                continue
            elif now_go_real == False:
                continue



        ########################################################################################
        # Determination of "thisFrameSize" of Non-dummy Frame Part Starts Here. 
        suggestedFrameSize = -np.Infinity 

        switch_to_AM = False 

        if (estimatingType == "ProbabilityPredict"):
            if now_go_real: 
                assert runningTime <= frame_prepared_times[singleFrame] + 1/FPS + pBufferTime or np.abs(frame_prepared_times[singleFrame] + 1/FPS + pBufferTime - runningTime)<=0.00001
                # print("---frame No." + str(singleFrame)+" start. genTime: "+str(frame_prepared_times[singleFrame])+" Now time is: "+str(runningTime))
            backLen = FPS * backSeconds
            timeSlot = frame_prepared_times[singleFrame] + 2/FPS + pBufferTime - runningTime # time allocation for transmission of a frame

            if (sum(transmitHistoryTimeLog) >= cut_off_time1):
                lookbackwardHistogramS =  utils.generatingBackwardSizeFromLog_fixLen(
                                            pastDurations= transmitHistoryTimeLog,
                                            pastDurationsCum= transmitHistoryTimeCum,
                                            pastSizes= realVideoFrameSize, 
                                            backLen= backLen,
                                            timeSlot= timeSlot, )

            else: lookbackwardHistogramS = []
            
            if (len(lookbackwardHistogramS)>0):

                loglookbackwardHistogramS = np.log(np.array(lookbackwardHistogramS))

                arg = 0
                try:    arg = np.argmax(pacf(np.array(loglookbackwardHistogramS))[1:])
                except:     arg = 0                    
                arg = arg + 1

                if now_go_real: effectCount += 1
                # print("len of lookbackwardHistogramS: " + str(len(lookbackwardHistogramS)))
                Shat_iMinus1 = loglookbackwardHistogramS[-1*arg]
                need_index = utils.extract_nearest_M_values_index(loglookbackwardHistogramS, Shat_iMinus1, M)
                need_index = np.array(need_index)
                need_index_plus_arg = need_index + arg
                decision_list = [loglookbackwardHistogramS[a] for a in need_index_plus_arg if a < len(loglookbackwardHistogramS)]

                if (now_go_real and len(percentiles)>=cut_off_time2*FPS):
                    controlled_epsilon = np.quantile(percentiles, pEpsilon, method="median_unbiased")
                else:
                    controlled_epsilon = pEpsilon

                suggestedFrameSize = np.exp(quantile(decision_list, controlled_epsilon, method='median_unbiased'))
                count_Cond_AlgoTimes += 1
                if (now_go_real): cumPartSize += suggestedFrameSize

                maxData = utils.calMaxData(prevTime=runningTime, 
                                        laterTime=runningTime+timeSlot, 
                                        packet_level_timestamp= networkEnvTime,
                                        packet_level_data= networkEnvPacket,)
                log_maxData = np.log(maxData)       
                percentiles.append( np.count_nonzero(decision_list <= log_maxData) / len(decision_list) )
                percentiles = percentiles[max(len(percentiles)-cut_off_time2 * FPS, 0) : ]

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

        countSize = True
        oldRunningTime = runningTime
        runningTime = uploadFinishTime
        # bufferTime = max(pBufferTime-max(runningTime-frame_prepared_times[singleFrame ],0), 0)  

        if (uploadFinishTime > frame_prepared_times[singleFrame] + 2/FPS + pBufferTime):    # encounter with frame dropping
            if (now_go_real):
                # print("要改時間了")
                countSize = False
                countExceed += 1            # countExceed retores the No. of skips in the 100 frames
                uploadDuration = frame_prepared_times[singleFrame] + 2/FPS + pBufferTime - oldRunningTime 
                runningTime = frame_prepared_times[singleFrame] + 2/FPS + pBufferTime
                # bufferTime = 0
                thisFrameSize = utils.calMaxData(prevTime=oldRunningTime, 
                                    laterTime=runningTime, 
                                    packet_level_timestamp= networkEnvTime,
                                    packet_level_data= networkEnvPacket,)


        throughputMeasure = thisFrameSize / uploadDuration
        throughputHistoryLog.append(throughputMeasure)
        transmitHistoryTimeLog.append(uploadDuration)
        realVideoFrameSize.append(thisFrameSize)

        if (len(transmitHistoryTimeCum)>0):
            transmitHistoryTimeCum.append(transmitHistoryTimeCum[-1]+uploadDuration)
        else:
            transmitHistoryTimeCum.append(uploadDuration)
        
        if (now_go_real and countSize):
            videoCumsize += thisFrameSize

        # Transmission of the Non-dummy Frame (whose size is determined above) Ends Here.
        ########################################################################################

        # will go to next "singleFrame"

    per100lr = exceedsRatios[1:]
    print( "Mean throughput in Mbps: " + str( videoCumsize/((singleFrame - startingFrame)/FPS) ) )
    print( "Mean of per100lr: " + str( mean(per100lr) ) + ", variance of per100lr: " + str(var(per100lr)))
    pyplot.plot(per100lr, color="blue")
    pyplot.xlabel("100 frames per slot")
    pyplot.ylabel("loss rate of that 100 frames")   
    pyplot.axhline(0.05, color="red")
    pyplot.legend(["real loss rate", "target 0.05"])
    pyplot.show()

    pyplot.hist(percentiles, bins=50, cumulative=True, density=True)
    pyplot.axline((0, 0), slope=1)
    pyplot.show()

    return [videoCumsize, failCount , minimal_framesize, frameCntAfterCutoff]



someMinimalFramesize = 0.005*1000/1024
someSubDummySize     = 0.005*1000/1024
# someInitialBuffer    = 0 
someInitialBuffer    = (3)*(1/FPS)                # cannot go with buffer now

statistics = uploadProcess(minimal_framesize= someMinimalFramesize, 
                                                    estimatingType = "ProbabilityPredict", 
                                                    pTrackUsed = 100, 
                                                    pBufferTime = someInitialBuffer)

failCount = statistics[1]
TotalTrials = statistics[3]

print("Fail count= " + str(failCount))
print("Overall loss rate=" + str(failCount/TotalTrials))