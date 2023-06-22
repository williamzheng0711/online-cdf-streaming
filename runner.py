import numpy as np
import padasip as pa
import utils
import matplotlib.pyplot as pyplot

from tqdm import tqdm
from statsmodels.tsa.stattools import acf, pacf
from optparse import OptionParser


#### Some constants.
howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024
FPS = 30                         # frame per second
minimal_framesize = 1e-7
M = 50


#### Some user input parameters. 
parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--algo", type="string", dest="algo", help="Which method to run? Choose ONE from 'OnCDF', 'OnRLS'.", default="OnCDF")
parser.add_option("--epsilon", type="float", dest="epsilon", help="Target frame loss rate (only useful for OnCDF and OnRLS)", default=0.05)
parser.add_option("--traceData", type="int", dest="traceData", help="Which trace data to simulate with? Input a number between 1 to 18", default=-1)
parser.add_option("--trainTime", type="int", dest="trainTime", help="How long (in seconds) is the training time interval in trace data?", default=1500)
parser.add_option("--testTime", type="int", dest="testTime", help="How long (in seconds) is the testing time interval in trace data?", default=1000)
parser.add_option("--Tb", type="float", dest="pBufferTime", help="player prefetch", default = -1 )
(options, args) = parser.parse_args()
algo = options.algo
assert algo in ["OnCDF", "OnRLS"]
traceData = options.traceData
assert traceData in range(19)
trainTime = options.trainTime                                      # number of active users
assert trainTime > 100 
testTime = options.testTime                                      # number of active users
assert testTime > 0
epsilon = options.epsilon
assert epsilon > 0 and epsilon < 1
pBufferTime = options.pBufferTime
assert pBufferTime > 0

print("Algo: " + algo +" Trace No.: " + str(traceData) +" trainTime= " + str(trainTime) + " testTime= "  + str(testTime) + " epsilon= "+ str(epsilon) + " Tb=" + str(pBufferTime))

### Read in the trace data. 
traceDir = './dataset/trace_data_simulator/'
# print("現在用的是dataset " + str(traceData) + " pBufferTime=" + str(pBufferTime))
count = 0
initialTime = 0
networkEnvTime = [] 
networkEnvPacket= [] 
with open(traceDir + str(traceData)+".txt") as traceDataFile:
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

c_hat_init = sum(networkEnvPacket[0:10000]) / (networkEnvTime[10000-1]-networkEnvTime[0]) # Just have a quick calc of the mean throughput
# print( str(c_hat_init) + "Mbps, this is mean throughput")


## Generate frame capture times
frame_capture_times = []       
tempTimeTrack = 0
fullTimeInSec = trainTime + testTime
for _ in range( fullTimeInSec * FPS ):
    tempTimeTrack = tempTimeTrack + 1/FPS
    frame_capture_times.append(tempTimeTrack) 


# variables that would be re-initialized every 100 frames
countExceed = 0                 # Integer. No. of exceeds in the this 100 frames

# variables that would get updated for all simulation
runningTime = 0                 # Float. The simulation clock, in seconds.
uploadDuration = 0              # Float. A variable to store the transmission time of frames.
videoCumsize = 0                # Float. A variable to store sum of all (non-loss) frames' size during the whole testing time period, i.e., after training time.
now_go_real = False             # Boolean of whether simulation clock >= training time 
effectCount = 0                 # Integer. Counts how many times our algorithm is applied on
failCount = 0                   # Integer. Counting how many frames are loss during the whole simulation in testing phase.
throughputHistoryLog = []       # Float array. Storing the avg. throughputs of every transmission (dummy and non-dummy)
realVideoFrameSize = []         # Float array. Storing sizes of every transmission (dummy and non-dummy)
transmitHistoryTimeLog = []     # Float array. Storing every transmission time (dummy and non-dummy)
transmitHistoryTimeCum = []     # Float array. CumSum of "transmitHistoryTimeLog"
exceedsRatios = []              # Float array. The i-th element is the loss ratio about the [100*i : 100*(i+1)]-th frames after training time. 
decision_list = []
percentiles = []
delays = []
errors = []
credibleLens = []
tuned_epsilon = epsilon
startingFrame = -1
debug = False

filt = pa.filters.FilterRLS(5, mu=0.99)  

### Transmit every frame, until we reach fullTimeInSec
for singleFrame in tqdm(range( fullTimeInSec * FPS )) if debug==False else range( fullTimeInSec * FPS ):
    assert runningTime - (frame_capture_times[singleFrame] + 1/FPS + pBufferTime) <= 1e-5

    # We want to know if we've trained for enough time
    if runningTime >= trainTime:
        if (frame_capture_times[singleFrame] < trainTime): continue
        now_go_real = True # Switch this variable on
        if startingFrame == -1:
            startingFrame = singleFrame
    
    # During testing phase, renew some variables/statistics per 100 frames
    if singleFrame % 100 == 0:
        message1 = "Frame: "+str(singleFrame)+". Now is time: "+str(runningTime)
        if now_go_real:
            message = message1 + " Exceed counts: " + str(countExceed)
            if algo == "OnCDF" and debug: 
                print(message+" Tuned epsilon: "+str(tuned_epsilon)) 
            elif algo == "OnRLS" and debug: 
                print(message) 
            exceedsRatios.append( countExceed/100 )
            countExceed = 0 
        elif now_go_real == False and debug:
            print(message1)

    # To return an end if has reached the terminus
    if runningTime > fullTimeInSec: 
        break
    
    # If have spare time, we do nothing but wait for the next event
    if runningTime < frame_capture_times[singleFrame]: runningTime = frame_capture_times[singleFrame]



    ########################################################################################
    # Determination of "thisFrameSize" of Non-dummy Frame Part Starts Here. 
    suggestedFrameSize = -np.Infinity 
    timeSlot = frame_capture_times[singleFrame] + 1/FPS + pBufferTime - runningTime # time allocation for transmission of a frame

    if (algo == "OnCDF"):
        backLen = FPS * M
        if len(transmitHistoryTimeLog) > 0:
            lookbackwardHistogramS =  utils.generatingBackwardSizeFromLog_fixLen(
                                        pastDurations= transmitHistoryTimeLog,
                                        pastDurationsCum= transmitHistoryTimeCum,
                                        pastSizes= realVideoFrameSize, 
                                        backLen= backLen,
                                        timeSlot= timeSlot)

        else: 
            lookbackwardHistogramS = []

        if (len(lookbackwardHistogramS)>M):
            if now_go_real: effectCount += 1
            loglookbackwardHistogramS = np.log(np.array(lookbackwardHistogramS))

            # arg = 0
            # try:    arg = np.argmax(pacf(np.array(loglookbackwardHistogramS))[1:])
            # except:     arg = 0                    
            # arg = arg + 1                
            # Shat_iMinus1 = loglookbackwardHistogramS[-1*arg]
            # need_index = utils.extract_nearest_M_values_index(loglookbackwardHistogramS, Shat_iMinus1, M)
            # need_index = np.array(need_index)
            # need_index_plus_arg = need_index + arg
            # decision_list = [loglookbackwardHistogramS[a] for a in need_index_plus_arg if a < len(loglookbackwardHistogramS)]

            decision_list = loglookbackwardHistogramS

            if (now_go_real and len(percentiles)>=backLen):
                tuned_epsilon = np.quantile(percentiles[:len(percentiles)-10], epsilon, method="interpolated_inverted_cdf")
            else:
                tuned_epsilon = epsilon

            credibleRegion90Len = np.exp(np.quantile(decision_list, 0.95)) - np.exp(np.quantile(decision_list, 0.05))
            # print("UL:" + str(np.exp(np.quantile(decision_list, 0.95))) +"  LL:" + str(np.exp(np.quantile(decision_list, 0.05))))
            credibleLens.append(credibleRegion90Len)

            suggestedFrameSize = np.exp(np.quantile(decision_list, tuned_epsilon))

            maxData = utils.calMaxData(prevTime=runningTime, 
                                    laterTime=runningTime+timeSlot, 
                                    packet_level_timestamp= networkEnvTime,
                                    packet_level_data= networkEnvPacket,)
            log_maxData = np.log(maxData)
            # maxDataSmall = utils.calMaxData(prevTime=runningTime, 
            #                         laterTime=runningTime+1/FPS, 
            #                         packet_level_timestamp= networkEnvTime,
            #                         packet_level_data= networkEnvPacket,)       
            # if now_go_real: print(singleFrame, maxData , maxDataSmall, suggestedFrameSize)

            percentiles.append( np.count_nonzero(decision_list <= log_maxData) / len(decision_list) )
            percentiles = percentiles[max(len(percentiles)-backLen, 0) : ]
        
        # else: 
        #     if now_go_real: 
        #         print(singleFrame, "WHAT??", len(lookbackwardHistogramS))
        #         break


    elif (algo == "OnRLS"):
        arr1 = np.array(np.array(realVideoFrameSize[-5:])/np.array(transmitHistoryTimeLog[-5:])) if (len(realVideoFrameSize)>=5 and len(transmitHistoryTimeLog)>=5) else np.ones(5) 
        # arr2 = np.array(realVideoFrameSize[-5:]) if len(realVideoFrameSize)>=5 else np.ones(5)
        # arr3 = np.array(transmitHistoryTimeLog[-5:]) if len(transmitHistoryTimeLog)>=5 else np.ones(5) 
        # arr3 = np.append(arr3, timeSlot)
        # input = np.concatenate((arr1, arr3))
        input = arr1
        # print(input)
        updatedX = input
                   
        c_avg_new_hat = filt.predict(updatedX) 
        pass
        error_quantile = np.quantile(errors[-1000:], epsilon) if len(errors)>0 else 0
        r_k = c_avg_new_hat + error_quantile
        # r_k = c_avg_new_hat

        credibleRegion90Len = timeSlot*(np.quantile(errors[-1000:], 0.95) - np.quantile(errors[-1000:], 0.05)) if len(errors)>0 else 0
        
        credibleLens.append(credibleRegion90Len)
        suggestedFrameSize = timeSlot * r_k
        # maxData = utils.calMaxData(prevTime=runningTime, 
        #                             laterTime=runningTime+timeSlot, 
        #                             packet_level_timestamp= networkEnvTime,
        #                             packet_level_data= networkEnvPacket,)


    if suggestedFrameSize == -np.Infinity:
        suggestedFrameSize = c_hat_init * timeSlot


    thisFrameSize =  max ( suggestedFrameSize, minimal_framesize ) # above is case-wise, now it is general to transmit a (video) frame

    # Determination of "thisFrameSize" of Non-dummy Frame Part Ends Here. 
    #######################################################################################



    ########################################################################################
    # Transmission of the Frame (whose size is determined done above) starts Here. 
    uploadFinishTime = utils.paper_frame_upload_finish_time(runningTime,networkEnvPacket,networkEnvTime, thisFrameSize)[0]
    uploadDuration = uploadFinishTime - runningTime; 

    countSize = True
    oldRunningTime = runningTime
    runningTime = uploadFinishTime

    if (uploadFinishTime > frame_capture_times[singleFrame] + 1/FPS + pBufferTime):    # encounter with frame defective
        countSize = False
        countExceed = (countExceed + 1)  if now_go_real  else countExceed        # countExceed retores the No. of skips in the 100 frames
        uploadDuration = frame_capture_times[singleFrame] + 1/FPS + pBufferTime - oldRunningTime 
        runningTime = frame_capture_times[singleFrame] + 1/FPS + pBufferTime
        thisFrameSize = utils.calMaxData(oldRunningTime,runningTime,networkEnvTime,networkEnvPacket)

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
        if  runningTime - frame_capture_times[singleFrame] > 1/FPS + pBufferTime:
            print(runningTime, frame_capture_times[singleFrame])
        delays.append( runningTime - frame_capture_times[singleFrame]) # playback time (transmission finish time) - capture time

    if algo == "OnRLS": 
        errors.append(thisFrameSize / uploadDuration - c_avg_new_hat)
        # print(singleFrame, thisFrameSize / uploadDuration - c_avg_new_hat)
        filt.adapt(thisFrameSize / uploadDuration, updatedX)

    # Transmission of the Frame Ends Here.
    ########################################################################################
    # will go to next "singleFrame"

# pyplot.hist(errors,bins=1000, range=(-50,50))
# pyplot.show()
per100lr = exceedsRatios[1:]
maxThroughputAll =  utils.calMaxData(startingFrame*(1/FPS), runningTime, networkEnvTime, networkEnvPacket)

JBurn = int(len(credibleLens)/2)
# pyplot.plot(credibleLens[JBurn:])
# pyplot.show()

print( "Mean of credible lengths: " + str(np.mean(credibleLens[JBurn:])))
print( "Mean throughput in Mbps: " + str( videoCumsize/(runningTime- startingFrame/FPS) ))
print( "Avg. Bandwidth in Mbps: " + str( maxThroughputAll/(runningTime- startingFrame/FPS) ))
print( "Mean of per100lr: " + str( np.mean(per100lr)) )
print( "Average delay time (about non-defective): " + str( np.mean(delays) ) + " seconds" )

# pyplot.title("Dataset " + str(traceData))
# pyplot.plot(delays, color="blue")
# pyplot.axhline(pBufferTime, color="red")
# pyplot.legend(["real delay", "maxBuffer"])
# pyplot.show()

# pyplot.title("Dataset " + str(traceData))
# pyplot.plot(per100lr, color="blue")
# pyplot.xlabel("100 frames per slot")
# pyplot.ylabel("loss rate of that 100 frames")   
# pyplot.axhline(epsilon, color=str(epsilon))
# pyplot.legend(["real loss rate", "target 0.05"])
# pyplot.show()

# pyplot.hist(percentiles, bins=50, cumulative=True, density=True)
# pyplot.axline((0, 0), slope=1)
# pyplot.show()






print("Done!")