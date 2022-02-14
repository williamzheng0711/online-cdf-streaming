import math
from warnings import catch_warnings
from matplotlib.pyplot import summer
from numpy.core.fromnumeric import argmax, partition
from numpy.lib.function_base import median
from scipy.stats import norm
from scipy.stats import laplace
import numpy as np
from numpy.core.fromnumeric import argmax, mean, size, var
import math
import random
import bisect

B_IN_MB = 1000.0*1000.0

def index_of_first(cdn_arrive_time, playingTime, pRealFrameNo ):
    for i,v in enumerate(cdn_arrive_time):
        if ( cdn_arrive_time[i+1] > playingTime and  cdn_arrive_time[i] <= playingTime    ):
            return i
    return None


def index_first_equal(number,list ):
    for i,v in enumerate(list):
        if (  v == number  ):
            return i
    return None


def frame_upload_done_time( runningTime, networkEnvBW, size, networkSamplingInterval ):
    shift = math.floor( runningTime/networkSamplingInterval )
    i = 0 

    while (size > 0): 
        if (i == 0):
            i = 1
            s_temp = size - networkEnvBW[shift]  *( networkSamplingInterval*(shift+1) - runningTime )
            # print(s_temp)
            if (s_temp<=0):
                t_out = (size / (networkEnvBW[shift] ) ) + runningTime
                return t_out
            size = s_temp

        else: 
            # print(shift)
            s_temp = size - networkEnvBW[shift]*networkSamplingInterval
            # print(s_temp)
            if (s_temp<=0): 
                t_out = networkSamplingInterval*shift + size / (networkEnvBW[shift] )
                return t_out
            size = s_temp

        shift = shift +1
    
    if (size<=0): 
        return runningTime
    print(size)


def find_gt_index(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError


def packet_level_frame_upload_finish_time( runningTime, packet_level_data, packet_level_timestamp, framesize ):
    # shift = next(x for x, val in enumerate(packet_level_timestamp) if val > runningTime) 
    shift = find_gt_index(a= packet_level_timestamp, x= runningTime)
    i = 0 

    while (framesize > 0): 
        if (i == 0):
            i = 1
            s_temp = framesize - packet_level_data[shift]
            # print(s_temp)
            if (s_temp<=0):
                t_out = packet_level_timestamp[shift]
                return t_out
            framesize = s_temp
        else: 
            # print(shift)
            s_temp = framesize - packet_level_data[shift]
            # print(s_temp)
            if (s_temp<=0): 
                t_out = packet_level_timestamp[shift]
                return t_out
            framesize = s_temp
        shift = shift +1

def expectationFunction(binsMe,probability,past,marginal):
    result = 0
    p1 = random.uniform(0, 1)
    p2 = 1-p1
    occur = sum(probability[past])
    prob = [ x/occur for x in probability[past] ]
    for p,index in zip(prob, range(len(prob)-1) ):
        # if (index >= marginal and index < len(binsMe)-marginal):
            result = result + p*(p1*binsMe[index]+p2*binsMe[index+1])
    
    return [result]

def expectationFunction2Past(binsMe,probability,past1, past2,marginal):
    result = 0
    p1 = random.uniform(0, 1)
    p2 = 1-p1
    occur = sum(probability[past2][past1])
    prob = [ x/occur for x in probability[past2][past1] ]
    for p,index in zip(prob, range(len(prob)-1) ):
        # if (index >= marginal and index < len(binsMe)-marginal):
            result = result + p*( p1 * binsMe[index] + p2 * binsMe[index+1])
    
    return [result]

def medianFunction(binsMe,probability,past):
    histogram_establish = []
    for ppValue, index in zip(probability[past], range(len(probability[past]) -1 )  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
    
    giveBackValue = laplace.fit(histogram_establish)[0]

    return [ giveBackValue, histogram_establish]


def mleFunction(binsMe,probability,past):
    histogram_establish = []
    for ppValue, index in zip(probability[past], range(len(probability[past]) -1 )  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
    
    giveBackValue = binsMe[np.argmax(probability[past])]

    return [ giveBackValue, histogram_establish]

def mleFunction2Past(binsMe,probability,past1,past2):
    histogram_establish = []
    p1 = random.uniform(0, 1)
    p2 = 1-p1
    for ppValue, index in zip(probability[past2][past1], range(len(probability[past2][past1]) -1 )  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            histogram_establish.append(p1*binsMe[index]+p2*binsMe[index+1])
    
    giveBackValue = binsMe[np.argmax(probability[past2][past1])]

    return [ giveBackValue, histogram_establish]


def veryConfidentFunction(binsMe,probability,past, quant):
    histogram_establish = []
    for ppValue, index in zip(probability[past], range(len(probability[past])-1)  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            # histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
            histogram_establish.append( binsMe[index] + (binsMe[index+1]-binsMe[index])*counter/ppValue )
    
    try: valueGiven = np.quantile(histogram_establish,quant)
    except: valueGiven = -1

    return [ valueGiven, histogram_establish]


def veryConfidentFunction2Past(binsMe,probability,past1,past2, quant):
    histogram_establish = []
    for ppValue, index in zip(probability[past2][past1], range(len(probability[past2][past1])-1)  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            # histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
            histogram_establish.append( binsMe[index] + (binsMe[index+1]-binsMe[index])*counter/ppValue )
    
    valueGiven = np.quantile(histogram_establish,quant)

    return [ valueGiven, histogram_establish]



def constructProbabilityModel(networkEnvBW, binsMe, networkSampleFreq, traceDataSampleFreq, threshold): 
        if (len(networkEnvBW)>threshold):
            networkEnvBW = networkEnvBW[len(networkEnvBW)-1 - threshold :len(networkEnvBW)-1]
        tobeDeleted = []
        probability  = [ [0] * len(binsMe)  for _ in range(len(binsMe))]
        marginalProbability  = [ 0 for  _ in  range(len(binsMe)) ]

        for j in range( 1,  len(networkEnvBW) ):
            if (j%1000 == 0) : print( str( (j/ len(networkEnvBW))*100 ) + "%, please wait."  )
            
            if (j % math.floor(traceDataSampleFreq/networkSampleFreq) == 0 ):
                self = 10
                past = 10

                if (j == 0 ):
                    continue

                for indexSelf in range(len(binsMe)-1): 
                    if (binsMe[indexSelf] <= networkEnvBW[j] and binsMe[indexSelf+1] > networkEnvBW[j]):
                        self = indexSelf
                
                for indexPast in range(len(binsMe)-1):
                    if (binsMe[indexPast] <= networkEnvBW[j-1] and binsMe[indexPast+1] > networkEnvBW[j-1]):
                        past = indexPast

                probability[past][self] += 1
                marginalProbability[self] += 1
                tobeDeleted.append(past)
                tobeDeleted.append(self)

        model = probability
    
        return [model,tobeDeleted]



def decideSamplingInterval(index):
#     8 is 0.25s interval
# 9 is 0.01s interval
# 10 is 0.1s interval 
# 13 is 0.02s interval


# The followings are all lower quality
# 11 is 0.1 3HK. 
# 12 is 0.04 interval. This is a network all values are in middle - low part 


# 14 is 1s
# 15 is 3s
    if (index == 8): return 0.25
    if (index == 9): return 0.01
    if (index == 10): return 0.1
    if (index == 11): return 0.1
    if (index == 12): return 0.04
    if (index == 13): return 0.02
    if (index == 14): return 1
    if (index == 15): return 3



