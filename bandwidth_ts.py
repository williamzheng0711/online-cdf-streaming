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
# parser.add_option("--traceData", type="int", dest="traceData", help="Which trace data to simulate with? Input a number between 1 to 3", default=-1)
parser.add_option("--trainTime", type="int", dest="trainTime", help="How long (in seconds) is the training time interval in trace data?", default=150)
parser.add_option("--testTime", type="int", dest="testTime", help="How long (in seconds) is the testing time interval in trace data?", default=1000)
(options, args) = parser.parse_args()

trainTime = options.trainTime                                     
assert trainTime > 100 
testTime = options.testTime                                     
assert testTime > 0

### Read in the trace data 1. 
traceDir1 = './dataset/trace_data_simulator/'
count1 = 0
initialTime1 = 0
networkEnvTime1 = [] 
networkEnvPacket1= [] 
with open(traceDir1 + str(1)+".txt") as traceDataFile:
    for line in traceDataFile:
        parse = line.split()
        if (count1==0):
            initialTime1 = float(parse[0])
        fileTime = float(parse[0]) 
        if (len(networkEnvTime1)>0 and fileTime - initialTime1 != networkEnvTime1[-1]): # common cases
            networkEnvTime1.append(fileTime - initialTime1)
            networkEnvPacket1.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
        elif (len(networkEnvTime1)==0):
            networkEnvTime1.append(fileTime - initialTime1)
            networkEnvPacket1.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
        else: # deal with packets with the same timestamp
            networkEnvPacket1[-1] += float(parse[1]) / howmany_Bs_IN_1Mb 
        count1 = count1  +1 

print("please wait...")
networkEnvBandwidth1 = Parallel(n_jobs=-1)(delayed(utils.calMaxData)(j,j+1,networkEnvTime1,networkEnvPacket1) for j in tqdm(range(trainTime, trainTime+testTime))) 



### Read in the trace data 2. 
traceDir2 = './dataset/trace_data_simulator/'
count2 = 0
initialTime2 = 0
networkEnvTime2 = [] 
networkEnvPacket2= [] 
with open(traceDir2 + str(2)+".txt") as traceDataFile:
    for line in traceDataFile:
        parse = line.split()
        if (count2==0):
            initialTime2 = float(parse[0])
        fileTime = float(parse[0]) 
        if (len(networkEnvTime2)>0 and fileTime - initialTime2 != networkEnvTime2[-1]): # common cases
            networkEnvTime2.append(fileTime - initialTime2)
            networkEnvPacket2.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
        elif (len(networkEnvTime2)==0):
            networkEnvTime2.append(fileTime - initialTime2)
            networkEnvPacket2.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
        else: # deal with packets with the same timestamp
            networkEnvPacket2[-1] += float(parse[1]) / howmany_Bs_IN_1Mb 
        count2 = count2  +1 

print("please wait...")
networkEnvBandwidth2 = Parallel(n_jobs=-1)(delayed(utils.calMaxData)(j,j+1,networkEnvTime2,networkEnvPacket2) for j in tqdm(range(trainTime, trainTime+testTime))) 



### Read in the trace data 3. 
traceDir3 = './dataset/trace_data_simulator/'
count3 = 0
initialTime3 = 0
networkEnvTime3 = [] 
networkEnvPacket3= [] 
with open(traceDir3 + str(3)+".txt") as traceDataFile:
    for line in traceDataFile:
        parse = line.split()
        if (count3==0):
            initialTime3 = float(parse[0])
        fileTime = float(parse[0]) 
        if (len(networkEnvTime3)>0 and fileTime - initialTime3 != networkEnvTime3[-1]): # common cases
            networkEnvTime3.append(fileTime - initialTime3)
            networkEnvPacket3.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
        elif (len(networkEnvTime3)==0):
            networkEnvTime3.append(fileTime - initialTime3)
            networkEnvPacket3.append( float(parse[1]) / howmany_Bs_IN_1Mb ) 
        else: # deal with packets with the same timestamp
            networkEnvPacket3[-1] += float(parse[1]) / howmany_Bs_IN_1Mb 
        count3 = count3  +1 

print("please wait...")
networkEnvBandwidth3 = Parallel(n_jobs=-1)(delayed(utils.calMaxData)(j,j+1,networkEnvTime3,networkEnvPacket3) for j in tqdm(range(trainTime, trainTime+testTime))) 



pyplot.plot(networkEnvBandwidth1)
pyplot.plot(networkEnvBandwidth2)
pyplot.plot(networkEnvBandwidth3)
pyplot.xlabel("Time (in second)")
pyplot.ylabel("Avg. bandwidth for each second")
pyplot.legend(["Trace No. 1", "Trace No. 2", "Trace No. 3"])
# pyplot.title("Avg. bandwidth in Mbps for Trace No. 1")
pyplot.show()




x_axis1 = np.linspace(min(networkEnvBandwidth1), max(networkEnvBandwidth1))
ecdf1 = sm.distributions.ECDF(networkEnvBandwidth1)
y_axis1 = ecdf1(x_axis1)

x_axis2 = np.linspace(min(networkEnvBandwidth2), max(networkEnvBandwidth2))
ecdf2 = sm.distributions.ECDF(networkEnvBandwidth2)
y_axis2 = ecdf2(x_axis2)

x_axis3 = np.linspace(min(networkEnvBandwidth3), max(networkEnvBandwidth3))
ecdf3 = sm.distributions.ECDF(networkEnvBandwidth3)
y_axis3 = ecdf3(x_axis3)

pyplot.xlabel("Bandwidth (in Mbps)")
pyplot.step(x_axis1, y_axis1)
pyplot.step(x_axis2, y_axis2)
pyplot.step(x_axis3, y_axis3)
pyplot.legend(["Trace No. 1", "Trace No. 2", "Trace No. 3"])
pyplot.show()
