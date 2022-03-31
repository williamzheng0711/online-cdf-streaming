from struct import pack
from numpy import sort


B_IN_MB = 1024*1024

whichVideo = 2
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
packet_level_integral_C = []
packet_level_time = []

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
                if (len(packet_level_integral_C)==0):
                    packet_level_integral_C.append(0 )
                else:
                    packet_level_integral_C.append(networkEnvPacket[-1]+packet_level_integral_C[-1])
                packet_level_time.append(nowFileTime - initialTime)
                count = count  +1 
            else:
                networkEnvTime.append(nowFileTime - initialTime)
                networkEnvPacket.append( float(parse[1]) / B_IN_MB ) 
                count = count  +1 
                
print("Before using time-packet, is time order correct? "+ str(packet_level_time == sorted(packet_level_time)))

print(packet_level_time[0:10])