import numpy as np
import padasip as pa
from optparse import OptionParser
import pmdarima as pm
from matplotlib import pyplot
from tqdm import tqdm


howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024
whichVideo = 2
epsilon = 0.1

parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--algo", type="string", dest="algo", help="Which method to run? Choose ONE from 'OnRLS', 'ARIMA'.", default="OnRLS")
(options, args) = parser.parse_args()

algo = options.algo
assert algo in ["ARIMA", "OnRLS"]


traceDir = './fixed_interval_tracedata/'
networkEnvPacket= [] 
for sufNum in range(whichVideo, whichVideo+1):
    with open( traceDir + str(sufNum) + ".csv" ) as traceDataFile:
        for line in traceDataFile:
            parse = line.split()
            networkEnvPacket.append( float(parse[0]) / howmany_Bs_IN_1Mb ) 

# print(networkEnvPacket[5])


startIndex = 500
testingSize = 60

errors = []
loss = 0
realSizeCum = 0
maxPossibleSize = 0

## OnRLS
pastLen = 5
arimaPastLen = 400
filt = pa.filters.FilterRLS(pastLen, mu=0.999)
guesses = []

for runningIndex in tqdm(range(startIndex, startIndex + testingSize)):
    if algo == "OnRLS":
        last5 = np.array(networkEnvPacket[runningIndex-pastLen:runningIndex])
        c_avg_new_hat = filt.predict(last5) 
        pass
        error_quantile = np.quantile(errors[-1000:], epsilon, method="median_unbiased") if len(errors)>0 else 0
        guess = max(c_avg_new_hat + error_quantile,0)
        guesses.append(guess)
        gap = networkEnvPacket[runningIndex] - guess
        errors.append(gap)
        filt.adapt(networkEnvPacket[runningIndex], last5)

    elif algo == "ARIMA":
        arima = pm.auto_arima(networkEnvPacket[:runningIndex], 
                            # error_action='ignore', 
                              trace=False,
                              suppress_warnings=True, 
                              maxiter=50, 
                              seasonal=True)
        guess = arima.predict(n_periods = 1)[0]
        guesses.append(guess)
        gap = networkEnvPacket[runningIndex] - guess
    
    if gap < 0:
        loss += 1
    else: 
        realSizeCum += guess
    
    maxPossibleSize += networkEnvPacket[runningIndex]


print("actual over-estimate rate", loss / testingSize)
print("actual bandwidth utilization", realSizeCum / maxPossibleSize)

if algo == "OnRLS":
    pyplot.plot(range(startIndex, startIndex + testingSize), guesses, 'o-')
    pyplot.plot(range(startIndex, startIndex + testingSize), networkEnvPacket[startIndex:startIndex + testingSize], 'o-')
    pyplot.legend(["OnRLS Decision", "Ground truth"])
    pyplot.show()

elif algo == "ARIMA":
    pyplot.plot(range(startIndex, startIndex + testingSize), guesses, 'o-')
    pyplot.plot(range(startIndex, startIndex + testingSize), networkEnvPacket[startIndex:startIndex + testingSize], 'o-')
    pyplot.legend(["ARIMA Predict", "Ground truth"])
    pyplot.xlabel("Index of time-intervals (each length 0.04 sec)")
    pyplot.title("How ARIMA perform in predicting?")
    pyplot.show()