import numpy as np
import padasip as pa
import pmdarima as pm
import utils
from matplotlib import pyplot
from tqdm import tqdm
from optparse import OptionParser
from joblib import Parallel, delayed
import statsmodels.api as sm # recommended import according to the docs


howmany_Bs_IN_1Mb = 1024*1024/8  # 1Mb = 1/8 MB = 1/8*1024*1024

parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--traceData", type="int", dest="traceData", help="Which trace data to simulate with? Input a number between 1 to 3", default=-1)
parser.add_option("--trainTime", type="int", dest="trainTime", help="How long (in seconds) is the training time interval in trace data?", default=150)
parser.add_option("--testTime", type="int", dest="testTime", help="How long (in seconds) is the testing time interval in trace data?", default=1000)
(options, args) = parser.parse_args()

traceData = options.traceData
assert traceData in range(19)
trainTime = options.trainTime                                     
assert trainTime > 100 
testTime = options.testTime                                     
assert testTime > 0

### Read in the trace data. 
traceDir = './dataset/trace_data_simulator/'
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
networkEnvBandwidth = Parallel(n_jobs=-1)(delayed(utils.calMaxData)(j,j+1,networkEnvTime,networkEnvPacket) for j in tqdm(range(trainTime, trainTime+testTime))) 

pyplot.plot(networkEnvBandwidth)
pyplot.xlabel("Time (in second)")
pyplot.ylabel("Avg. bandwidth for each second")
# pyplot.title("Avg. bandwidth in Mbps for Trace No. 1")
pyplot.show()

x_axis = np.linspace(min(networkEnvBandwidth), max(networkEnvBandwidth))
ecdf = sm.distributions.ECDF(networkEnvBandwidth)
y_axis = ecdf(x_axis)
pyplot.xlabel("Bandwidth (in Mbps)")
# pyplot.title("Empirical CDF of bandwidth in Mbps for Tr")
pyplot.step(x_axis, y_axis)
pyplot.show()
