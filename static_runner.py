import numpy as np
import padasip as pa


howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024
whichVideo = 2
epsilon = 0.1

traceDir = './fixed_interval_tracedata/'
networkEnvPacket= [] 
for sufNum in range(whichVideo, whichVideo+1):
    with open( traceDir + str(sufNum) + ".csv" ) as traceDataFile:
        for line in traceDataFile:
            parse = line.split()
            networkEnvPacket.append( float(parse[0]) / howmany_Bs_IN_1Mb ) 

# print(networkEnvPacket[5])


startIndex = 100
testingSize = 50000

errors = []
loss = 0
pastLen = 10
filt = pa.filters.FilterRLS(pastLen, mu=0.999)
realSizeCum = 0
maxPossibleSize = 0

for runningIndex in range(startIndex, startIndex + testingSize):
    last5 = np.array(networkEnvPacket[runningIndex-pastLen:runningIndex])
    c_avg_new_hat = filt.predict(last5) 
    pass

    error_quantile = np.quantile(errors[-700:], epsilon, method="median_unbiased") if len(errors)>0 else 0
    guess = max(c_avg_new_hat + error_quantile,0)

    gap = networkEnvPacket[runningIndex] - guess
    errors.append(gap)
    filt.adapt(networkEnvPacket[runningIndex], last5)

    maxPossibleSize += networkEnvPacket[runningIndex]

    print("error_quantile", error_quantile, "gap", gap)

    if gap < 0:
        loss += 1
    else: 
        realSizeCum += c_avg_new_hat



print(loss / testingSize)
print(realSizeCum / maxPossibleSize)