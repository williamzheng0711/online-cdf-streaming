
from cmath import sqrt
from random import sample
from statistics import mean, variance

from numpy import quantile


B_IN_MB = 1024*1024

whichVideo = 6
FPS = 30

# Testing Set Size
howLongIsVideoInSeconds = 60

# Training Data Size
timePacketsDataLoad = 4000000

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
            else:
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / B_IN_MB ) 
                count = count  +1 
                
print("Before using time-packet, is time order correct? "+ str(packet_level_time_training == sorted(packet_level_time_training, reverse=False)))
print("Before using integral records, is the order correct? "+ str(packet_level_integral_C_training == sorted(packet_level_integral_C_training, reverse=False)))
# All things above are  "environment"'s initialization, 
# which cannot be manipulated by our algorithm.
############################################################################
# All things below are of our business



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

print(mean(sampleThroughputRecord))
print(sqrt(variance(sampleThroughputRecord,pEpsilon)))

t_experiment = (mean(sampleThroughputRecord)/quantile(sampleThroughputRecord,pEpsilon)-1)/FPS

print(t_experiment)