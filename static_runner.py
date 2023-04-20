import numpy as np
import padasip as pa
import pmdarima as pm
import utils
from matplotlib import pyplot
from tqdm import tqdm
from optparse import OptionParser
from joblib import Parallel, delayed


howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024

parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--algo", type="string", dest="algo", help="Which method to run? Choose ONE from 'OnRLS', 'ARIMA'.", default="OnRLS")
parser.add_option("--epsilon", type="float", dest="epsilon", help="Target frame loss rate (only useful for OnABC and OnRLS)", default=0.05)
(options, args) = parser.parse_args()

algo = options.algo
assert algo in ["ARIMA", "OnRLS"]
epsilon = options.epsilon
assert epsilon > 0 and epsilon < 1


FPS = 30
startIndex = 300
testingSize = 3000
traceData = 18

### Read in the trace data. 

traceDir = './dataset/fyp_lab/'
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

print("please wait...")
networkEnvBandwidth = Parallel(n_jobs=-1)(delayed(utils.calMaxData)(j*1/FPS,(j+1)*1/FPS,networkEnvTime,networkEnvPacket) for j in tqdm(range(startIndex+testingSize))) 

errors = []
loss = 0
realSizeCum = 0
maxPossibleSize = 0

## ARIMA
arimaPastLen = 100

## OnRLS
pastLen = 5
filt = pa.filters.FilterRLS(pastLen, mu=0.999)

guesses = []
now_go_real = False

for runningIndex in tqdm(range(0, startIndex + testingSize)):
    if runningIndex == startIndex:
        now_go_real = True

    if algo == "OnRLS":
        last5 = np.array(networkEnvBandwidth[runningIndex-pastLen:runningIndex]) if runningIndex > pastLen else np.zeros(pastLen)
        c_avg_new_hat = filt.predict(last5) 
        pass
        error_quantile = np.quantile(errors[-1000:], epsilon, method="median_unbiased") if len(errors)>0 else 0
        guess = max(c_avg_new_hat + error_quantile,0)
        guesses.append(guess)
        gap = networkEnvBandwidth[runningIndex] - c_avg_new_hat
        errors.append(gap)
        filt.adapt(networkEnvBandwidth[runningIndex], last5)

    elif algo == "ARIMA":
        if runningIndex > 2:
            arima = pm.auto_arima(networkEnvBandwidth[:runningIndex], 
                                # error_action='ignore', 
                                trace=False,
                                suppress_warnings=True, 
                                maxiter=50, 
                                seasonal=True)
            guess = arima.predict(n_periods = 1)[0]
            guesses.append(guess)
            gap = networkEnvBandwidth[runningIndex] - guess
        else:
            guess = 1
            guesses.append(guess)
    
    if now_go_real:
        if networkEnvBandwidth[runningIndex] - guess < 0:
            loss += 1
        else: 
            realSizeCum += guess
        maxPossibleSize += networkEnvBandwidth[runningIndex]


print("actual over-estimate rate", loss / testingSize)
print("actual bandwidth utilization", realSizeCum / maxPossibleSize)

if algo == "OnRLS":
    pyplot.plot(range(startIndex, startIndex + testingSize), guesses[startIndex:startIndex + testingSize], 'o-')
    pyplot.plot(range(startIndex, startIndex + testingSize), networkEnvBandwidth[startIndex:startIndex + testingSize], 'o-')
    pyplot.legend(["OnRLS Decision", "Ground truth"])
    pyplot.show()

    pyplot.hist(errors, bins=50)
    pyplot.show()

elif algo == "ARIMA":
    pyplot.plot(range(startIndex, startIndex + testingSize), guesses[startIndex:startIndex + testingSize], 'o-')
    pyplot.plot(range(startIndex, startIndex + testingSize), networkEnvBandwidth[startIndex:startIndex + testingSize], 'o-')
    pyplot.legend(["ARIMA Predict", "Ground truth"])
    pyplot.xlabel("Index of time-intervals (each length 0.04 sec)")
    pyplot.title("How ARIMA perform in predicting?")
    pyplot.show()