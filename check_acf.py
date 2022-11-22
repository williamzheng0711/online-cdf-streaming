from statsmodels.graphics.tsaplots import *
import matplotlib.pyplot as pyplot

whichVideo = 9

howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024


traceDir = './dataset/network_trace/1h_less/'
count = 0
initialTime = 0
networkEnvTime = [] 
networkEnvPacket= [] 
for sufNum in range(whichVideo, whichVideo+1):
    with open( traceDir + str(sufNum) + ".csv" ) as traceDataFile:
        for line in traceDataFile:
            parse = line.split()
            networkEnvPacket.append(float(parse[0])/howmany_Bs_IN_1Mb)
            # parse = line.split()
            # if (count==0):
            #     initialTime = float(parse[0])
            # fileTime = float(parse[0]) 
            # if (len(networkEnvTime)>0 and fileTime - initialTime != networkEnvTime[-1]): # common cases
            #     networkEnvTime.append(fileTime - initialTime)
            #     networkEnvPacket.append( float(parse[1]) ) 
            # elif (len(networkEnvTime)==0):
            #     networkEnvTime.append(fileTime - initialTime)
            #     networkEnvPacket.append( float(parse[1]) ) 
            # else: # deal with packets with the same timestamp
            #     networkEnvPacket[-1] += float(parse[1]) 
            # count = count  +1 

test_array = networkEnvPacket[1200:4200]
# print(test_array)
plot_pacf(test_array, lags = 20)
pacf_array = pacf(test_array, nlags = 20)
print(pacf_array)
pyplot.show()
